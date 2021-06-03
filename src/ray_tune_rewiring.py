import argparse
import os
import time
from functools import partial
import json
import numpy as np
import torch
import torch.nn.functional as F
from data import get_dataset, set_train_val_test_split
from GNN_early import GNNEarly
from GNN import GNN, MP
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.ax import AxSearch
from run_GNN import get_optimizer, test, test_OGB, train, train_OGB
from torch import nn
from GNN_ICML20 import ICML_GNN, get_sym_adj
from GNN_ICML20 import train as train_icml
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from run_best_ray import get_best_params_dir
from graph_rewiring import apply_gdc, KNN, apply_KNN, apply_beltrami, apply_edge_sampling
from graph_rewiring_ray import set_search_space, set_rewiring_space, set_cora_search_space, set_citeseer_search_space
from run_best_ray_rewiring_top5 import top5
"""
python3 ray_tune.py --dataset ogbn-arxiv --lr 0.005 --add_source --function transformer --attention_dim 16 --hidden_dim 128 --heads 4 --input_dropout 0 --decay 0 --adjoint --adjoint_method rk4 --method rk4 --time 5.08 --epoch 500 --num_samples 1 --name ogbn-arxiv-test --gpus 1 --grace_period 50 

"""


def average_test(models, datas, pos_encoding):
  results = [test(model, data, pos_encoding) for model, data in zip(models, datas)]
  train_accs, val_accs, tmp_test_accs = [], [], []

  for train_acc, val_acc, test_acc in results:
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    tmp_test_accs.append(test_acc)

  return train_accs, val_accs, tmp_test_accs


def average_test_OGB(models, mps, datas, pos_encoding):
  results = [test_OGB(model, mp, data, pos_encoding, opt) for model, mp, data in zip(models, mps, datas)]
  train_accs, val_accs, tmp_test_accs = [], [], []

  for train_acc, val_acc, test_acc in results:
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    tmp_test_accs.append(test_acc)

  return train_accs, val_accs, tmp_test_accs


def train_ray_rand(opt, checkpoint_dir=None, data_dir="../data"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # dataset = get_dataset(opt, data_dir, opt['not_lcc'])

  models = []
  mps = []
  datas = []
  optimizers = []

  for split in range(opt["num_splits"]):
    dataset = get_dataset(opt, data_dir, opt['not_lcc'])
    dataset.data = set_train_val_test_split(
      np.random.randint(0, 1000), dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
    # datas.append(dataset.data)

    if opt['beltrami'] and opt['dataset'] == 'ogbn-arxiv':
      pos_encoding = apply_beltrami(dataset.data, opt, data_dir=data_dir)
      mp = MP(opt, pos_encoding.shape[1], device=torch.device('cpu'))
    elif opt['beltrami']:
      pos_encoding = apply_beltrami(dataset.data, opt, data_dir=data_dir).to(device)
      opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
      pos_encoding = None

    data = dataset.data.to(device)
    datas.append(data)

    if opt['baseline']:
      opt['num_feature'] = dataset.num_node_features
      opt['num_class'] = dataset.num_classes
      adj = get_sym_adj(dataset.data, opt, device)
      model, data = ICML_GNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
      train_this = train_icml
    else:
      # model = GNN(opt, dataset, device)
      # if opt['rewire_KNN'] or opt['edge_sampling']:
      if opt['rewire_KNN'] or opt['KNN_online'] or opt['edge_sampling'] or opt['edge_sampling_online']:
        model = GNN_KNN(opt, dataset, device).to(device)
      else:
        model = GNN(opt, dataset, device).to(device)
      train_this = train

    model = model.to(device)
    models.append(model)

    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)

    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
    optimizers.append(optimizer)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
      checkpoint = os.path.join(checkpoint_dir, "checkpoint")
      model_state, optimizer_state = torch.load(checkpoint)
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      KNN_ei = [apply_KNN(data, pos_encoding, model, opt) for model, data in zip(models, datas)]
      for i, data in enumerate(datas):
        models[i].odeblock.odefunc.edge_index = KNN_ei[i]
    if opt['edge_sampling'] and epoch % opt['edge_sampling_epoch'] == 0 and epoch != 0:
      apply_edge_sampling(data.x, pos_encoding, model, opt)

    if opt['dataset'] == 'ogbn-arxiv':
      loss = np.mean([train_OGB(model, mp, optimizer, data, pos_encoding) for model, optimizer, data in
                      zip(models, mps, optimizers, datas)])
      train_accs, val_accs, tmp_test_accs = average_test_OGB(models, mps, datas, pos_encoding)
    else:
      loss = np.mean(
        [train_this(model, optimizer, data, pos_encoding) for model, optimizer, data in zip(models, optimizers, datas)])
      train_accs, val_accs, tmp_test_accs = average_test(models, datas, pos_encoding)

    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      best = np.argmax(val_accs)
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
    tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs), train_acc=np.mean(train_accs),
                forward_nfe=model.fm.sum,
                backward_nfe=model.bm.sum)


def train_ray(opt, checkpoint_dir=None, data_dir="../data"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, opt['not_lcc'])

  models = []
  mps = []
  optimizers = []

  if opt['beltrami'] and opt['dataset'] == 'ogbn-arxiv':
    pos_encoding = apply_beltrami(dataset.data, opt, data_dir=data_dir)
    mp = MP(opt, pos_encoding.shape[1], device=torch.device('cpu'))
  elif opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt, data_dir=data_dir).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  dataset.data = dataset.data.to(device)
  datas = [dataset.data for i in range(opt["num_init"])]

  for split in range(opt["num_init"]):
    if opt['baseline']:
      opt['num_feature'] = dataset.num_node_features
      opt['num_class'] = dataset.num_classes
      adj = get_sym_adj(dataset.data, opt, device)
      model, data = ICML_GNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
      train_this = train_icml
    else:
      # if opt['rewire_KNN'] or opt['edge_sampling']:
      if opt['rewire_KNN'] or opt['KNN_online'] or opt['edge_sampling'] or opt['edge_sampling_online']:
        model = GNN_KNN(opt, dataset, device).to(device)
      else:
        model = GNN(opt, dataset, device).to(device)
      train_this = train

      data = dataset.data.to(device)

    models.append(model)

    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)

    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
    optimizers.append(optimizer)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
      checkpoint = os.path.join(checkpoint_dir, "checkpoint")
      model_state, optimizer_state = torch.load(checkpoint)
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      ei = apply_KNN(data.x, pos_encoding, model, opt)
      model.odeblock.odefunc.edge_index = ei

    if opt['edge_sampling'] and epoch % opt['edge_sampling_epoch'] == 0 and epoch != 0:
      apply_edge_sampling(data.x, pos_encoding, model, opt)

    if opt['dataset'] == 'ogbn-arxiv':
      loss = np.mean([train_OGB(model, mp, optimizer, data, pos_encoding) for model, optimizer, data in
                      zip(models, mps, optimizers, datas)])
      train_accs, val_accs, tmp_test_accs = average_test_OGB(models, mps, datas, pos_encoding)
    else:
      loss = np.mean(
        [train_this(model, optimizer, data, pos_encoding) for model, optimizer, data in zip(models, optimizers, datas)])
      train_accs, val_accs, tmp_test_accs = average_test(models, datas, pos_encoding)

    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      best = np.argmax(val_accs)
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
    tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs), train_acc=np.mean(train_accs),
                forward_nfe=model.fm.sum,
                backward_nfe=model.bm.sum)


def train_ray_int(opt, checkpoint_dir=None, data_dir="../data"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, opt['not_lcc'])

  if opt["num_splits"] > 0:
    dataset.data = set_train_val_test_split(
      23 * np.random.randint(0, opt["num_splits"]),  # random prime 23 to make the splits 'more' random. Could remove
      dataset.data,
      num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  if opt['beltrami'] and opt['dataset'] == 'ogbn-arxiv':
    pos_encoding = apply_beltrami(dataset.data, opt, data_dir=data_dir)
    mp = MP(opt, pos_encoding.shape[1], device=torch.device('cpu'))
  elif opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt, data_dir=data_dir).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  # if opt['rewire_KNN'] or opt['edge_sampling']:
  if opt['rewire_KNN'] or opt['KNN_online'] or opt['edge_sampling'] or opt['edge_sampling_online']:
    model = GNN_KNN(opt, dataset, device) if opt["no_early"] else GNNKNNEarly(opt, dataset, device)
  else:
    model = GNN(opt, dataset, device) if opt["no_early"] else GNNEarly(opt, dataset, device)

  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  model, data = model.to(device), dataset.data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])

  if checkpoint_dir:
    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
  best_time = best_epoch = train_acc = val_acc = test_acc = 0
  for epoch in range(1, opt["epoch"]):
    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      ei = apply_KNN(data.x, pos_encoding, model, opt)
      model.odeblock.odefunc.edge_index = ei

    if opt['edge_sampling'] and epoch % opt['edge_sampling_epoch'] == 0 and epoch != 0:
      apply_edge_sampling(data.x, pos_encoding, model, opt)

    loss = train(model, optimizer, data, pos_encoding)

    if opt["no_early"]:
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)
      best_time = opt['time']
      if tmp_val_acc > val_acc:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        test_acc = tmp_test_acc
    else:
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)
      if tmp_val_acc > val_acc:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        test_acc = tmp_test_acc
      if model.odeblock.test_integrator.solver.best_val > val_acc:
        best_epoch = epoch
        val_acc = model.odeblock.test_integrator.solver.best_val
        test_acc = model.odeblock.test_integrator.solver.best_test
        train_acc = model.odeblock.test_integrator.solver.best_train
        best_time = model.odeblock.test_integrator.solver.best_time
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)
    tune.report(loss=loss, accuracy=val_acc, test_acc=test_acc, train_acc=train_acc, best_time=best_time,
                best_epoch=best_epoch,
                forward_nfe=model.fm.sum, backward_nfe=model.bm.sum)


def main(opt):
  data_dir = os.path.abspath("../data")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  opt = set_search_space(opt)
  scheduler = ASHAScheduler(
    metric=opt['metric'],
    mode="max",
    max_t=opt["epoch"],
    grace_period=opt["grace_period"],
    reduction_factor=opt["reduction_factor"],
  )
  reporter = CLIReporter(
    metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe", "backward_nfe"]
  )
  # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
  search_alg = AxSearch(metric=opt['metric'])
  search_alg = None

  train_fn = train_ray if opt["num_splits"] == 0 else train_ray_rand

  result = tune.run(
    partial(train_fn, data_dir=data_dir),
    name=opt["name"],
    resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
    search_alg=search_alg,
    keep_checkpoints_num=3,
    checkpoint_score_attr=opt['metric'],
    config=opt,
    num_samples=opt["num_samples"],
    scheduler=scheduler,
    max_failures=2,
    local_dir="../ray_tune",
    progress_reporter=reporter,
    raise_on_failed_trial=False,
  )


def mainLoop(opt):
  opt['epoch'] = 250
  opt['gpus'] = 1
  opt['num_samples'] = 400
  opt['grace_period'] = 20
  opt['reduction_factor'] = 10
  opt['num_splits'] = 4
  # opt['num_init'] = 3
  opt['max_nfe'] = 3000

  opt['rewiring'] = None #'pos_enc_knn' #'gdc' #None
  opt['beltrami'] = True
  opt['adjoint'] = False
  opt['rewire_KNN'] = False
  opt['edge_sampling'] = False
  opt['fa_layer'] = False

  datas = ['CoauthorCS'] #['Cora', 'Citeseer']
  folders = ['CoauthorCS_final_tune_posencGDC'] #['Cora_beltrami_5split','Citeseer_beltrami_5split']
  for i, ds in enumerate(datas):
    print(f"Running Tuning for {ds}")
    opt["dataset"] = ds
    opt["name"] = folders[i]
    main(opt)


def edge_sampling_online_space(opt):
  opt['edge_sampling_online'] = True
  opt['edge_sampling_add_type'] = 'gdc' #tune.choice(['importance','random'])
  # opt['edge_sampling_space'] = tune.choice(['attention', 'pos_distance', 'z_distance'])
  opt['edge_sampling_online_reps'] = tune.choice([2,3,4])
  opt['edge_sampling_sym'] = False #tune.choice([True, False])
  # opt['edge_sampling_add'] = tune.choice([0.04, 0.08, 0.16, 0.32, 0.64]) # tune.choice([0.04, 0.08, 0.16, 0.32])
  # opt['edge_sampling_rmv'] = tune.choice([0.0, 0.04, 0.08])  # tune.choice([0.04, 0.08, 0.16, 0.32])
  opt["time"] = tune.uniform(0.25, 5.0)
  return opt

def fa_layer_space(opt):
  opt['fa_layer'] = True
  opt['fa_layer_time'] = tune.choice([1, 2, 3])  # 1.0
  opt['fa_layer_method'] = 'rk4'
  opt['fa_layer_step_size'] = tune.choice([0.5, 1])  # 1.0
  opt['fa_layer_edge_sampling_rmv'] = tune.choice([0, 0.25, 0.5, 0.75])
  return opt

def top5_onlineSampling_FAlayer(opt):
  data_dir = os.path.abspath("../data")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  opt['max_nfe'] = 3000
  opt['num_samples'] = 100
  opt['grace_period'] = 20
  opt['reduction_factor'] = 10
  opt['epoch'] = 250
  opt['num_splits'] = 3
  opt['gpus'] = 1
  opt['no_early'] = True
  opt['metric'] = 'test_acc'


  opt['block'] = 'attention'
  opt['function'] = 'laplacian'
  opt['beltrami'] = True
  opt['edge_sampling'] = False
  opt['rewire_KNN'] = False

  datas = ['Cora','Citeseer']
  idxs = [0,1,2,3,4]

  ###Getting the best params from random sources
  best_Cora_params = top5
  best_Citeseer_params = []
  for i in idxs:
    CiteseerOpt = {'folder':'Citeseer_beltrami_1','index':4,'metric':'test_acc'}
    Citeseer_best_params_dir = get_best_params_dir(CiteseerOpt)
    with open(Citeseer_best_params_dir + '/params.json') as f:
      best_Citeseer_param = json.loads(f.read())
    best_Citeseer_params.append(best_Citeseer_param)
  best_params_each = [best_Cora_params, best_Citeseer_params]

  #edge sampling
  ESnames = ['Cora_onlineSampling_no_early', 'Citeseer_onlineSampling_no_early']
  opt['edge_sampling_online'] = True
  opt['fa_layer'] = False
  for i, (data, best_params) in enumerate(zip(datas, best_params_each)):
    for idx in idxs:
      opt['dataset'] = data
      opt['index'] = idxs[i]
      name = ESnames[i]
      opt['name'] = name
      best_params = best_params_each[i][idx]
      best_params_ret = {**best_params, **opt}
      best_params_ret = edge_sampling_online_space(best_params_ret)

      try:
        best_params_ret['mix_features']
      except KeyError:
        best_params_ret['mix_features'] = False
      try:
        best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
      except:
        pass
      print("Running with parameters {}".format(best_params_ret))


      scheduler = ASHAScheduler(
        metric=best_params_ret['metric'],
        mode="max",
        max_t=opt["epoch"],
        grace_period=best_params_ret["grace_period"],
        reduction_factor=best_params_ret["reduction_factor"],
      )
      reporter = CLIReporter(
        metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe", "backward_nfe"]
      )
      # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
      search_alg = AxSearch(metric=best_params_ret['metric'])
      search_alg = None

      train_fn = train_ray if best_params_ret["num_splits"] == 0 else train_ray_rand

      result = tune.run(
        partial(train_fn, data_dir=data_dir),
        name=best_params_ret["name"],
        resources_per_trial={"cpu": best_params_ret["cpus"], "gpu": best_params_ret["gpus"]},
        search_alg=search_alg,
        keep_checkpoints_num=3,
        checkpoint_score_attr=opt['metric'],
        config=best_params_ret,
        num_samples=best_params_ret["num_samples"],
        scheduler=scheduler,
        max_failures=2,
        local_dir="../ray_tune",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
      )

  #FA layer
  FAnames = ['Cora_FAlayer_no_early', 'Citeseer_FAlayer_no_early']
  opt['edge_sampling_online'] = False
  opt['fa_layer'] = True
  for i, (data, best_params) in enumerate(zip(datas, best_params_each)):
    for idx in idxs:
      opt['dataset'] = data
      opt['index'] = idxs[i]
      name = FAnames[i]
      opt['name'] = name
      best_params = best_params_each[i][idx]
      best_params_ret = {**best_params, **opt}
      best_params_ret = fa_layer_space(best_params_ret)

      try:
        best_params_ret['mix_features']
      except KeyError:
        best_params_ret['mix_features'] = False
      try:
        best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
      except:
        pass
      print("Running with parameters {}".format(best_params_ret))

      scheduler = ASHAScheduler(
        metric=best_params_ret['metric'],
        mode="max",
        max_t=best_params_ret["epoch"],
        grace_period=best_params_ret["grace_period"],
        reduction_factor=best_params_ret["reduction_factor"],
      )
      reporter = CLIReporter(
        metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe", "backward_nfe"]
      )
      # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
      search_alg = AxSearch(metric=best_params_ret['metric'])
      search_alg = None

      train_fn = train_ray if best_params_ret["num_splits"] == 0 else train_ray_rand

      result = tune.run(
        partial(train_fn, data_dir=data_dir),
        name=best_params_ret["name"],
        resources_per_trial={"cpu": best_params_ret["cpus"], "gpu": best_params_ret["gpus"]},
        search_alg=search_alg,
        keep_checkpoints_num=3,
        checkpoint_score_attr=best_params_ret['metric'],
        config=best_params_ret,
        num_samples=best_params_ret["num_samples"],
        scheduler=scheduler,
        max_failures=2,
        local_dir="../ray_tune",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
      )

def ES_test(opt):
  data_dir = os.path.abspath("../data")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  opt['max_nfe'] = 3000
  opt['num_samples'] = 20
  opt['grace_period'] = 20
  opt['reduction_factor'] = 10
  opt['epoch'] = 250
  opt['num_splits'] = 3
  opt['gpus'] = 1
  opt['no_early'] = True
  opt['metric'] = 'test_acc'

  opt['block'] = 'attention'
  opt['function'] = 'laplacian'
  opt['beltrami'] = True
  opt['edge_sampling'] = False
  opt['rewire_KNN'] = False
  opt['edge_sampling_online'] = True
  opt['fa_layer'] = False

  datas = ['Cora', 'Citeseer']
  idxs = [0, 1, 2, 3, 4]

  ###Getting the best params from random sources
  best_Cora_params = top5
  best_Citeseer_params = []
  for i in idxs:
    CiteseerOpt = {'folder': 'Citeseer_beltrami_1', 'index': 4, 'metric': 'test_acc'}
    Citeseer_best_params_dir = get_best_params_dir(CiteseerOpt)
    with open(Citeseer_best_params_dir + '/params.json') as f:
      best_Citeseer_param = json.loads(f.read())
    best_Citeseer_params.append(best_Citeseer_param)
  best_params_each = [best_Cora_params, best_Citeseer_params]

  # edge sampling
  ESnames = ['Cora_onlineSampling_2', 'Citeseer_onlineSampling_2']
  for i, (data, best_params) in enumerate(zip(datas, best_params_each)):
    for idx in idxs:
      # best_params[idx]['time'] = best_params[idx]['time'] / 3
      opt['dataset'] = data
      opt['index'] = idxs[i]
      name = ESnames[i]
      opt['name'] = name
      best_params = best_params_each[i][idx]
      best_params_ret = {**best_params, **opt}

      best_params_ret['edge_sampling_online'] = True
      best_params_ret['edge_sampling_add_type'] = tune.choice(['importance', 'random'])
      best_params_ret['edge_sampling_space'] = 'attention' #tune.choice(['attention', 'pos_distance', 'z_distance'])
      best_params_ret['edge_sampling_online_reps'] = 3 #tune.choice([2, 3, 4])
      # best_params_ret['edge_sampling_sym'] = False #tune.choice([True, False])
      best_params_ret['edge_sampling_add'] = tune.choice([0.04, 0.08, 0.16, 0.32, 0.64])  # tune.choice([0.04, 0.08, 0.16, 0.32])
      best_params_ret['edge_sampling_rmv'] = tune.choice([0.0, 0.04, 0.08])  # tune.choice([0.04, 0.08, 0.16, 0.32])
      best_params_ret["time"] = best_params_ret["time"] / 3

      # best_params_ret["time"] = tune.uniform(0.25, 5.0)

      # best_params_ret = edge_sampling_online_space(best_params_ret)

      try:
        best_params_ret['mix_features']
      except KeyError:
        best_params_ret['mix_features'] = False
      try:
        best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
      except:
        pass
      print("Running with parameters {}".format(best_params_ret))

      scheduler = ASHAScheduler(
        metric=best_params_ret['metric'],
        mode="max",
        max_t=opt["epoch"],
        grace_period=best_params_ret["grace_period"],
        reduction_factor=best_params_ret["reduction_factor"],
      )
      reporter = CLIReporter(
        metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe",
                        "backward_nfe"]
      )
      # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
      search_alg = AxSearch(metric=best_params_ret['metric'])
      search_alg = None

      train_fn = train_ray if best_params_ret["num_splits"] == 0 else train_ray_rand

      result = tune.run(
        partial(train_fn, data_dir=data_dir),
        name=best_params_ret["name"],
        resources_per_trial={"cpu": best_params_ret["cpus"], "gpu": best_params_ret["gpus"]},
        search_alg=search_alg,
        keep_checkpoints_num=3,
        checkpoint_score_attr=opt['metric'],
        config=best_params_ret,
        num_samples=best_params_ret["num_samples"],
        scheduler=scheduler,
        max_failures=2,
        local_dir="../ray_tune",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
      )


def pos_enc_DW_tune(opt):
  opt['epoch'] = 250
  opt['gpus'] = 1
  # opt['num_samples'] = 250
  num_samples = [250, 100]
  opt['grace_period'] = 20
  opt['reduction_factor'] = 10
  opt['num_splits'] = 4
  # opt['num_init'] = 3
  opt['max_nfe'] = 3000

  opt['rewiring'] = 'pos_enc_knn' #'gdc' #None
  opt['beltrami'] = True
  opt['adjoint'] = False
  opt['rewire_KNN'] = False
  opt['edge_sampling'] = False
  opt['fa_layer'] = False

  datas = ['Citeseer','Photo']#['Cora', 'Citeseer', 'Photo', 'Computers', 'CoauthorCS', 'Pubmed']
  folders = ['Citeseer_DW_rewiring4','Photo_DW_rewiring5'] #['Cora_DW_rewiring','Citeseer_DW_rewiring','Photo_DW_rewiring',
              # 'Computers_DW_rewiring','CoauthorCS_DW_rewiring','Pubmed_DW_rewiring']
  for i, ds in enumerate(datas):
    opt['num_samples'] = num_samples[i]
    print(f"Running Tuning for {ds}")
    opt["dataset"] = ds
    opt["name"] = folders[i]
    main(opt)

def pos_enc_gdc_tune(opt):
  opt['epoch'] = 250
  opt['gpus'] = 1
  # opt['num_samples'] = 250
  num_samples = [100, 50]
  opt['grace_period'] = 20
  opt['reduction_factor'] = 10
  opt['num_splits'] = 4
  # opt['num_init'] = 3
  opt['max_nfe'] = 3000

  opt['rewiring'] = 'pos_enc_knn' #'gdc' #None
  opt['beltrami'] = True
  opt['adjoint'] = False
  opt['rewire_KNN'] = False
  opt['edge_sampling'] = False
  opt['fa_layer'] = False

  datas = ['Citeseer','Photo']#['Cora', 'Citeseer', 'Photo', 'Computers', 'CoauthorCS', 'Pubmed']
  folders = ['Citeseer_gdc_rewiring','Photo_gdc_rewiring'] #['Cora_DW_rewiring','Citeseer_DW_rewiring','Photo_DW_rewiring',
              # 'Computers_DW_rewiring','CoauthorCS_DW_rewiring','Pubmed_DW_rewiring']
  for i, ds in enumerate(datas):
    opt['num_samples'] = num_samples[i]
    print(f"Running Tuning for {ds}")
    opt["dataset"] = ds
    opt["name"] = folders[i]
    main(opt)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--use_cora_defaults",
    action="store_true",
    help="Whether to run with best params for cora. Overrides the choice of dataset",
  )
  parser.add_argument(
    "--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
  parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension.")
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument("--input_dropout", type=float, default=0.5, help="Input dropout rate.")
  parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer.")
  parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
  parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay for optimization")
  parser.add_argument("--self_loop_weight", type=float, default=1.0, help="Weight of self-loops.")
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs per iteration.")
  parser.add_argument("--alpha", type=float, default=1.0, help="Factor in front matrix A.")
  parser.add_argument("--time", type=float, default=1.0, help="End time of ODE function.")
  parser.add_argument("--augment", action="store_true",
                      help="double the length of the feature vector by appending zeros to stabilise ODE learning", )
  parser.add_argument("--alpha_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) alpha")
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument("--beta_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) beta")
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                      help='Add a fully connected layer to the encoder.')

  # ODE args
  parser.add_argument(
    "--method", type=str, default="dopri5", help="set the numerical solver: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--step_size', type=float, default=1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun",
    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                      help="use the adjoint ODE method to reduce memory footprint")
  parser.add_argument("--tol_scale", type=float, default=1.0, help="multiplier for atol and rtol")
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument("--ode_blocks", type=int, default=1, help="number of ode blocks to run")
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  # SDE args
  parser.add_argument("--dt_min", type=float, default=1e-5, help="minimum timestep for the SDE solver")
  parser.add_argument("--dt", type=float, default=1e-3, help="fixed step size")
  parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
  # Attention args
  parser.add_argument(
    "--leaky_relu_slope",
    type=float,
    default=0.2,
    help="slope of the negative part of the leaky relu used in attention",
  )
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
  parser.add_argument("--attention_norm_idx", type=int, default=0, help="0 = normalise rows, 1 = normalise cols")
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  # ray args
  parser.add_argument("--num_samples", type=int, default=20, help="number of ray trials")
  parser.add_argument("--gpus", type=float, default=0, help="number of gpus per trial. Can be fractional")
  parser.add_argument("--cpus", type=float, default=1, help="number of cpus per trial. Can be fractional")
  parser.add_argument(
    "--grace_period", type=int, default=5, help="number of epochs to wait before terminating trials"
  )
  parser.add_argument(
    "--reduction_factor", type=int, default=4, help="number of trials is halved after this many epochs"
  )
  parser.add_argument("--name", type=str, default="ray_exp")
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
  parser.add_argument("--num_init", type=int, default=1, help="Number of random initializations >= 0")

  parser.add_argument("--max_nfe", type=int, default=300, help="Maximum number of function evaluations allowed.")
  parser.add_argument('--metric', type=str, default='accuracy', help='metric to sort the hyperparameter tuning runs on')
  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  parser.add_argument("--baseline", action="store_true", help="Wheather to run the ICML baseline or not.")
  parser.add_argument("--regularise", dest='regularise', action='store_true', help='search over reg params')

  # rewiring args
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="above this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
  parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
  parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
  parser.add_argument('--threshold_type', type=str, default="addD_rvR", help="topk_adj, addD_rvR")
  parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
  parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
  parser.add_argument('--attention_rewiring', action='store_true',
                      help='perform DIGL using precalcualted GRAND attention')

  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--pos_enc_type', type=str, default="GDC", help='positional encoder (default: GDC)')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")

  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.05, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.05, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_space', type=str, default="pos_distance", help="pos_distance, z_distance")
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--symmetric_attention', action='store_true', help='maks the attention symmetric for rewring in QK space')

  parser.add_argument('--fa_layer', action='store_true', help='fully connected final ode')

  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,cosine_power,pearson,rank_pearson, exp_kernel_pos, exp_kernel_z")


  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

  args = parser.parse_args()
  opt = vars(args)
  # main(opt)
  mainLoop(opt)
  # top5_onlineSampling_FAlayer(opt)
  # ES_test(opt)
  # pos_enc_gdc_tune(opt)