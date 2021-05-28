import argparse
from ray.tune import Analysis
import json
import os
import pandas as pd
import numpy as np
from utils import get_sem, mean_confidence_interval
from ray_tune_rewiring import train_ray_int, main as main_ray
from ray import tune
from functools import partial
import os, time
from ray.tune import CLIReporter
from run_best_ray_rewiring_top5 import top5


def appendDFToCSV_void(df, csvFilePath, sep=","):
  import os
  if not os.path.isfile(csvFilePath):
    df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
  elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
    raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
      len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
  elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
    raise Exception("Columns and column order of dataframe and csv file do not match!!")
  else:
    df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

def get_best_params_dir(opt):
  analysis = Analysis("../ray_tune/{}".format(opt['folder']))
  df = analysis.dataframe(metric='accuracy', mode='max')
  best_params_dir = df.sort_values(opt['metric'], ascending=False)['logdir'].iloc[opt['index']]
  return best_params_dir

def with_KNN(opt):
  opt['rewire_KNN'] = True
  opt['rewire_KNN_T'] = "T0"
  opt['rewire_KNN_epoch'] = 20
  opt['rewire_KNN_k'] = 64
  opt['rewire_KNN_sym'] = False
  return opt


def run_best_params(opt):
  best_params_dir = get_best_params_dir(opt)
  with open(best_params_dir + '/params.json') as f:
    best_params = json.loads(f.read())
  # allow params specified at the cmd line to override
  best_params_ret = {**best_params, **opt}
  try:
    best_params_ret['mix_features']
  except KeyError:
    best_params_ret['mix_features'] = False
  # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
  best_params_ret['epoch'] = opt['epoch']
  best_params_ret['max_nfe'] = opt['max_nfe']
  # handle adjoint
  if best_params['adjoint'] or opt['adjoint']:
    best_params_ret['adjoint'] = True

  if opt["run_with_KNN"]:
    best_params_ret = with_KNN(best_params_ret)

  if opt['change_att_sim_type']:
    best_params_ret['attention_type'] = opt['att_sim_type']
    best_params_ret['square_plus'] = False

  try:
    best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
  except:
    pass

  print("Running with parameters {}".format(best_params_ret))

  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch", "training_iteration", "forward_nfe", "backward_nfe"])

  if opt['name'] is None:
    name = opt['folder'] + '_test'
  else:
    name = opt['name']

  result = tune.run(
    partial(train_ray_int, data_dir=data_dir),
    name=name,
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    search_alg=None,
    keep_checkpoints_num=3,
    checkpoint_score_attr='accuracy',
    config=best_params_ret,
    num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
    scheduler=None,
    max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

  try:
    csvFilePath = '../ray_results/{}.csv'.format(name)  # , time.strftime("%Y%m%d-%H%M%S"))
    appendDFToCSV_void(df, csvFilePath, sep=",")
    # df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
  except:
    pass

  print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

  test_accs = df['test_acc'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))


def mainLoop(opt):
  opt['max_nfe'] = 1000
  opt['epoch'] = 250
  opt['num_splits'] = 8
  opt['gpus'] = 1
  opt['earlystopxT'] = 2 #5
  opt['metric'] = 'test_acc'

  datas = ['Cora','Citeseer','Pubmed']#,'Cora','Citeseer']#['Cora']#,'Citeseer'] #, 'Pubmed'] #['Cora', 'Citeseer', 'Photo']
  folders = ['Cora_hyp_encs','Citeseer_hyp_encs'] #,'Cora_beltrami_exp_kernel_z2','Citeseer_beltrami_exp_kernel_z2']#['Cora_beltrami_exp_kernel'] #beltrami_2','Citeseer_beltrami_1']#, 'Pubmed_beltrami_2_KNN'] #['Cora_beltrami_1_KNN', 'Citeseer_beltrami_1_KNN', 'Photo_beltrami_1_KNN']
  names = ['Cora_hyp_encs','Citeseer_hyp_encs']#,'blah2','blah2']#['Citeseer_beltrami_exp_kernel_test'] #['Cora_beltrami_attdefaults_test','Citeseer_beltrami_attdefaults_test']
  indexes = [[0,1,2,3,4],[0,1,2,3,4]] #,[0,1,2,3,4],[0,1,2,3,4]] #,[3,4]] #[[0,1,2], [0,1,2]] #,3,4]]#, [0,1,2,3,4]] #,0,0]
  opt['run_with_KNN'] = False
  opt['bestwithKNN'] = False
  opt['edge_sampling'] = False
  opt['change_att_sim_type'] = False
  opt['bestwithAttTypes'] = ['cosine_sim', 'scaled_dot'] #[False]

  for i, ds in enumerate(datas):
    for idx_i, idx in enumerate(indexes[i]):
      if opt['change_att_sim_type']:
        for att_type in opt['bestwithAttTypes']:
          print(f"Running Best Params for {ds}")
          opt["dataset"] = ds
          opt["folder"] = folders[i]
          opt["name"] = f"{names[i]}{'_KNN' if opt['bestwithKNN'] else ''}"
          opt["index"] = indexes[i][idx_i]
          opt['att_sim_type'] = att_type
          run_best_params(opt)
      else:
        print(f"Running Best Params for {ds}")
        opt["dataset"] = ds
        opt["folder"] = folders[i]
        opt["name"] = f"{names[i]}{'_KNN' if opt['bestwithKNN'] else ''}"
        opt["index"] = indexes[i][idx_i]
        run_best_params(opt)

  # run_top5(opt) #with edge sampling



def KNN_abalation(opt):
  datas = ['Cora','Citeseer']
  folders = ['beltrami_2','Citeseer_beltrami_1']
  names = ['Cora_beltrami_ablation','Citeseer_beltrami_ablation']
  indexes = [[0],[0]]
  ks = [4,8,16,32,64]
  KNN_epochs = [2,5,10,25,50]
  TS = ['T0','TN']
  KNN_sysms = [True, False]
  opt['rewire_KNN'] = True
  opt['reps'] = 4 #8 #16
  opt['epoch'] = 100

  for i, ds in enumerate(datas):
    print(f"Running Best Params for {ds}")
    opt["dataset"] = ds
    opt["folder"] = folders[i]
    opt["name"] = f"{names[i]}"

    best_params_dir = get_best_params_dir(opt)
    with open(best_params_dir + '/params.json') as f:
      best_params = json.loads(f.read())
    best_params_ret = {**best_params, **opt} # allow params specified at the cmd line to override
    try:
      best_params_ret['mix_features']
    except KeyError:
      best_params_ret['mix_features'] = False

    best_params_ret['feat_hidden_dim'] = 64  #<----override default

    for idx_i, idx in enumerate(indexes[i]):
      best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
      best_params_ret["index"] = indexes[i][idx_i]
      for k in ks:
        best_params_ret['rewire_KNN_k'] = k
        for KNN_epoch in KNN_epochs:
          best_params_ret['rewire_KNN_epoch'] = KNN_epoch
          for t in TS:
            best_params_ret['rewire_KNN_T'] = t
            for tf_sym in KNN_sysms:
              best_params_ret['rewire_KNN_sym'] = tf_sym

              print("Running with parameters {}".format(best_params_ret))

              data_dir = os.path.abspath("../data")
              reporter = CLIReporter(
                metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch",
                                "training_iteration", "forward_nfe", "backward_nfe"])

              if opt['name'] is None:
                name = opt['folder'] + '_test'
              else:
                name = opt['name']

              result = tune.run(
                partial(train_ray_int, data_dir=data_dir),
                name=name,
                resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
                search_alg=None,
                keep_checkpoints_num=3,
                checkpoint_score_attr='accuracy',
                config=best_params_ret,
                num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
                scheduler=None,
                max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
                local_dir='../ray_tune',
                progress_reporter=reporter,
                raise_on_failed_trial=False)

              df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

              try:
                csvFilePath = '../ray_results/{}.csv'.format(name)  # , time.strftime("%Y%m%d-%H%M%S"))
                appendDFToCSV_void(df, csvFilePath, sep=",")
                # df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
              except:
                pass

              print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

              test_accs = df['test_acc'].values
              print("test accuracy {}".format(test_accs))
              log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
              print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs),
                               mean_confidence_interval(test_accs)))


def set_pos_enc_space(opt):
  opt['rewiring'] = False #tune.choice([True, False])
  opt['exact'] = True
  opt['feat_hidden_dim'] = 64  # <----override default

  opt['reweight_attention'] = tune.sample_from(lambda spec: True if spec.config.rewiring else False)
  opt['rewire_KNN'] = tune.sample_from(lambda spec: False if spec.config.rewiring else True)

  # opt['rewire_KNN'] = True
  opt['epoch'] = 100
  opt['num_init'] = 4
  opt['gpus'] = 1

  opt['rewire_KNN_k'] = tune.choice([4, 8, 16, 32, 64])
  opt['rewire_KNN_epoch'] = tune.choice([2, 5, 10, 25, 50])
  opt['rewire_KNN_T'] = tune.choice(['T0', 'TN'])
  opt['rewire_KNN_sym'] = tune.choice([True, False])

  return opt


def run_top5(opt):
  opt['name'] = 'Cora_top52_Esonline0_3'
  # opt['reps'] = 8
  opt['edge_sampling'] = False
  opt['rewire_KNN'] = False
  opt['KNN_online'] = False
  opt['symmetric_attention'] = False

  # # opt['edge_sampling_online'] = True #*** this might not have been hit
  # opt['edge_sampling_online_reps'] = 1
  # opt['edge_sampling_add_type'] = 'random'
  # opt['edge_sampling_space'] = 'attention'
  # opt['edge_sampling_add'] = 0
  # opt['att_samp_pct_rmv'] = 0

  opt['max_nfe'] = 2000
  opt['epoch'] = 200
  opt['num_splits'] = 8
  opt['gpus'] = 1
  opt['earlystopxT'] = 5
  opt['metric'] = 'test_acc'

  for idx, best_params in enumerate(top5):
    opt['index'] = idx
    best_params_ret = {**best_params, **opt}

    best_params_ret['edge_sampling_online'] = True #*** this might not have been hit
    best_params_ret['edge_sampling_online_reps'] = 1
    best_params_ret['edge_sampling_add_type'] = 'random'
    best_params_ret['edge_sampling_space'] = 'attention'
    best_params_ret['edge_sampling_add'] = 0
    best_params_ret['edge_sampling_rmv'] = 0

    try:
      best_params_ret['mix_features']
    except KeyError:
      best_params_ret['mix_features'] = False
    # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
    best_params_ret['epoch'] = opt['epoch']
    best_params_ret['max_nfe'] = opt['max_nfe']
    # handle adjoint
    if best_params['adjoint'] or opt['adjoint']:
      best_params_ret['adjoint'] = True

    if opt["run_with_KNN"]:
      best_params_ret = with_KNN(best_params_ret)

    if opt['change_att_sim_type']:
      best_params_ret['attention_type'] = opt['att_sim_type']
      best_params_ret['square_plus'] = False

    try:
      best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
    except:
      pass
    print("Running with parameters {}".format(best_params_ret))

    data_dir = os.path.abspath("../data")
    reporter = CLIReporter(
      metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch",
                      "training_iteration", "forward_nfe", "backward_nfe"])

    if opt['name'] is None:
      name = opt['folder'] + '_test'
    else:
      name = opt['name']

    result = tune.run(
      partial(train_ray_int, data_dir=data_dir),
      name=name,
      resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
      search_alg=None,
      keep_checkpoints_num=3,
      checkpoint_score_attr='accuracy',
      config=best_params_ret,
      num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
      scheduler=None,
      max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
      local_dir='../ray_tune',
      progress_reporter=reporter,
      raise_on_failed_trial=False)

    df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

    try:
      csvFilePath = '../ray_results/{}.csv'.format(name)  # , time.strftime("%Y%m%d-%H%M%S"))
      appendDFToCSV_void(df, csvFilePath, sep=",")
      # df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
    except:
      pass

    print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

    test_accs = df['test_acc'].values
    print("test accuracy {}".format(test_accs))
    log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
    print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs),
                     mean_confidence_interval(test_accs)))

def run_top_withES(opt):
  opt['max_nfe'] = 2000
  opt['epoch'] = 200
  opt['num_splits'] = 8
  opt['gpus'] = 1
  # opt['earlystopxT'] = 5
  opt['no_early'] = True
  opt['metric'] = 'test_acc'

  opt['name'] = 'Cora_top52_Esonline_varySampling2'
  # opt['reps'] = 8
  opt['edge_sampling'] = False
  opt['rewire_KNN'] = False
  opt['KNN_online'] = False
  opt['symmetric_attention'] = False
  opt['fa_layer'] = False

  opt['edge_sampling_online'] = True
  opt['edge_sampling_online_reps'] = 3
  opt['edge_sampling_add_type'] = 'random'
  opt['edge_sampling_space'] = 'attention'
  samples = [0, 0.01, 0.02, 0.04, 0.08]
  opt['edge_sampling_sym'] = False

  best_params = top5[0]
  best_params['time'] = best_params['time'] / 3
  # for idx, best_params in enumerate(top5):
  idx = 0
  for add in samples:
    for rmv in samples:
      best_params_ret = {**best_params, **opt}
      best_params_ret['edge_sampling_online'] = True
      best_params_ret['edge_sampling_online_reps'] = 3
      best_params_ret['edge_sampling_add_type'] = 'random'
      best_params_ret['edge_sampling_space'] = 'attention'
      best_params_ret['edge_sampling_add'] = add
      best_params_ret['edge_sampling_rmv'] = rmv
      best_params_ret['index'] = idx
      idx = idx + 1
      try:
        best_params_ret['mix_features']
      except KeyError:
        best_params_ret['mix_features'] = False
      # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
      best_params_ret['epoch'] = opt['epoch']
      best_params_ret['max_nfe'] = opt['max_nfe']
      # handle adjoint
      if best_params['adjoint'] or opt['adjoint']:
        best_params_ret['adjoint'] = True

      if opt["run_with_KNN"]:
        best_params_ret = with_KNN(best_params_ret)

      if opt['change_att_sim_type']:
        best_params_ret['attention_type'] = opt['att_sim_type']
        best_params_ret['square_plus'] = False

      try:
        best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
      except:
        pass
      print("Running with parameters {}".format(best_params_ret))

      data_dir = os.path.abspath("../data")
      reporter = CLIReporter(
        metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch",
                        "training_iteration", "forward_nfe", "backward_nfe"])

      if opt['name'] is None:
        name = opt['folder'] + '_test'
      else:
        name = opt['name']

      result = tune.run(
        partial(train_ray_int, data_dir=data_dir),
        name=name,
        resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
        search_alg=None,
        keep_checkpoints_num=3,
        checkpoint_score_attr='accuracy',
        config=best_params_ret,
        num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
        scheduler=None,
        max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
        local_dir='../ray_tune',
        progress_reporter=reporter,
        raise_on_failed_trial=False)

      df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

      try:
        csvFilePath = '../ray_results/{}.csv'.format(name)  # , time.strftime("%Y%m%d-%H%M%S"))
        appendDFToCSV_void(df, csvFilePath, sep=",")
        # df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
      except:
        pass

      print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

      test_accs = df['test_acc'].values
      print("test accuracy {}".format(test_accs))
      log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
      print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs),
                       mean_confidence_interval(test_accs)))

def KNN_abalation_grid(opt):
  opt["dataset"] = 'Cora'
  opt["folder"] = 'beltrami_2'
  opt["name"] = 'Cora_beltrami_KNN_ablation_grid'
  opt['index'] = 0

  best_params_dir = get_best_params_dir(opt)
  with open(best_params_dir + '/params.json') as f:
    best_params = json.loads(f.read())
  best_params_ret = {**best_params, **opt}  # allow params specified at the cmd line to override
  try:
    best_params_ret['mix_features']
  except KeyError:
    best_params_ret['mix_features'] = False

  best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']

  best_params_ret = set_pos_enc_space(best_params_ret)
  main_ray(best_params_ret)


def edge_sampling_ablation(opt):
  # folders = ['Cora_top5','Citeseer_beltrami_1']
  folders = ['Cora_edgeS_ablation','Citeseer_edgeS_ablation']
  datas = ['Cora','Citeseer']

  opt['edge_sampling'] = True
  opt['edge_sampling_T'] = 'TN'
  opt['edge_sampling_epoch'] = 2 #10
  # opt['edge_sampling_space'] = 'z_distance'
  # opt['edge_sampling_add'] = 0.16
  # opt['edge_sampling_rmv'] = 0.16

  # TI = ['T0','TN']
  sample_spaces = ['z_distance', 'pos_distance']
  samples = [0.0, 0.02, 0.04, 0.08, 0.16, 0.32]
  opt['edge_sampling_sym'] = False

  opt['max_nfe'] = 2000
  opt['epoch'] = 200
  opt['num_splits'] = 8
  opt['gpus'] = 1
  opt['earlystopxT'] = 5
  opt['no_early'] = True
  opt['metric'] = 'test_acc'


  ###Getting the best params from random sources
  best_Cora_params = top5[0]
  best_Citeseer_params = []
  idxs = [0]
  for i in idxs:
    CiteseerOpt = {'folder':'Citeseer_beltrami_1','index':4,'metric':'test_acc'}
    Citeseer_best_params_dir = get_best_params_dir(CiteseerOpt)
    with open(Citeseer_best_params_dir + '/params.json') as f:
      best_Citeseer_param = json.loads(f.read())
    best_Citeseer_params.append(best_Citeseer_param)

  best_params_each = [best_Cora_params, best_Citeseer_params]


  for i, (folder, data, best_params) in enumerate(zip(folders, datas, best_params_each)):
  # for i, (folder, data) in enumerate(zip(folders, datas)):
    opt['folder'] = folder
    opt['dataset'] = data
    opt['index'] = idxs[i]

    best_params_dir = get_best_params_dir(opt)
    with open(best_params_dir + '/params.json') as f:
      best_params = json.loads(f.read())

    # for ti in TI:
    #   opt['edge_sampling_T'] = ti
    for sample_space in sample_spaces:
      opt['edge_sampling_space'] = sample_space
      for add in samples:
        opt['edge_sampling_add'] = add

        for rmv in samples:
          opt['experiment'] = f"add_{add}_rmv_{rmv}"
          opt['edge_sampling_rmv'] = rmv

          best_params_ret = {**best_params, **opt}
          try:
            best_params_ret['mix_features']
          except KeyError:
            best_params_ret['mix_features'] = False
          # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
          best_params_ret['epoch'] = opt['epoch']
          best_params_ret['max_nfe'] = opt['max_nfe']
          # handle adjoint
          if best_params['adjoint'] or opt['adjoint']:
            best_params_ret['adjoint'] = True

          if opt["run_with_KNN"]:
            best_params_ret = with_KNN(best_params_ret)

          if opt['change_att_sim_type']:
            best_params_ret['attention_type'] = opt['att_sim_type']
            best_params_ret['square_plus'] = False

          try:
            best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
          except:
            pass
          print("Running with parameters {}".format(best_params_ret))

          data_dir = os.path.abspath("../data")
          reporter = CLIReporter(
            metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch",
                            "training_iteration", "forward_nfe", "backward_nfe"])

          if opt['name'] is None:
            name = opt['folder'] + '_test'
          else:
            name = opt['name']

          result = tune.run(
            partial(train_ray_int, data_dir=data_dir),
            name=name,
            resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
            search_alg=None,
            keep_checkpoints_num=3,
            checkpoint_score_attr='accuracy',
            config=best_params_ret,
            num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
            scheduler=None,
            max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
            local_dir='../ray_tune',
            progress_reporter=reporter,
            raise_on_failed_trial=False)

          df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

          try:
            csvFilePath = '../ray_results/{}.csv'.format(name)  # , time.strftime("%Y%m%d-%H%M%S"))
            appendDFToCSV_void(df, csvFilePath, sep=",")
            # df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
          except:
            pass

          print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

          test_accs = df['test_acc'].values
          print("test accuracy {}".format(test_accs))
          log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
          print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs),
                           mean_confidence_interval(test_accs)))


def edge_sampling_online_ablation(opt):
  # folders = ['Cora_edgeS_ablation','Citeseer_edgeS_ablation']
  # datas = ['Cora','Citeseer']
  folders = ['Citeseer_edgeS_ablation']
  datas = ['Citeseer']

  opt['max_nfe'] = 2000
  opt['epoch'] = 200
  opt['num_splits'] = 8
  opt['gpus'] = 1
  # opt['earlystopxT'] = 5
  opt['no_early'] = True
  opt['metric'] = 'test_acc'

  opt['edge_sampling'] = False
  opt['rewire_KNN'] = False
  opt['KNN_online'] = False
  opt['symmetric_attention'] = False
  opt['fa_layer'] = False
  opt['beltrami'] = True
  opt['pos_enc_type'] = 'GDC'

  opt['edge_sampling_online'] = True
  opt['edge_sampling_online_reps'] = 3
  # opt['edge_sampling_add_type'] = 'random'
  opt['edge_sampling_space'] = 'attention'
  samples = [0, 0.01, 0.02, 0.04, 0.08]
  sample_add_types = ['importance', 'random']
  opt['edge_sampling_sym'] = False

  ###Getting the best params from random sources
  # best_Cora_params = top5[0]
  best_Citeseer_params = []
  idxs = [0]
  for i in idxs:
    CiteseerOpt = {'folder':'Citeseer_beltrami_1','index':4,'metric':'test_acc'}
    Citeseer_best_params_dir = get_best_params_dir(CiteseerOpt)
    with open(Citeseer_best_params_dir + '/params.json') as f:
      best_Citeseer_param = json.loads(f.read())
    best_Citeseer_params.append(best_Citeseer_param)

  # best_params_each = [best_Cora_params, best_Citeseer_params]
  best_params_each = [best_Citeseer_params[0]]

  idx = 0
  for i, (folder, data, best_params) in enumerate(zip(folders, datas, best_params_each)):
    best_params['time'] = best_params['time'] / 3
    opt['folder'] = folder
    opt['dataset'] = data
    for add_type in sample_add_types:
      opt['edge_sampling_add_type'] = add_type
      for add in samples:
        opt['edge_sampling_add'] = add
        for rmv in samples:
          opt['edge_sampling_rmv'] = rmv
          opt['experiment'] = f"add_{add}_rmv_{rmv}"

          best_params_ret = {**best_params, **opt}
          best_params_ret['index'] = idx
          idx = idx + 1

          try:
            best_params_ret['mix_features']
          except KeyError:
            best_params_ret['mix_features'] = False
          # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
          best_params_ret['epoch'] = opt['epoch']
          best_params_ret['max_nfe'] = opt['max_nfe']
          # handle adjoint
          if best_params['adjoint'] or opt['adjoint']:
            best_params_ret['adjoint'] = True

          try:
            best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']
          except:
            pass
          print("Running with parameters {}".format(best_params_ret))

          data_dir = os.path.abspath("../data")
          reporter = CLIReporter(
            metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch",
                            "training_iteration", "forward_nfe", "backward_nfe"])

          result = tune.run(
            partial(train_ray_int, data_dir=data_dir),
            name=folder,
            resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
            search_alg=None,
            keep_checkpoints_num=3,
            checkpoint_score_attr='accuracy',
            config=best_params_ret,
            num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
            scheduler=None,
            max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
            local_dir='../ray_tune',
            progress_reporter=reporter,
            raise_on_failed_trial=False)



def embeddings_ablation(opt):
  opt['max_nfe'] = 1000
  opt['epoch'] = 400
  opt['num_splits'] = 8
  opt['gpus'] = 1
  opt['earlystopxT'] = 2 #5
  opt['metric'] = 'test_acc'

  datas = ['Photo','Citeseer']
  folders = ['Photo_DW_rewiring5','Citeseer_DW_rewiring4']
  names = ['Photo_DW_rewiring5_ablation','Photo_DW_rewiring5_ablation']#,'Citeseer_DW_rewiring4_ablation']
  indexes = [[1],[1]]
  opt['run_with_KNN'] = False
  opt['bestwithKNN'] = False
  opt['edge_sampling'] = False
  opt['change_att_sim_type'] = False
  opt['bestwithAttTypes'] = ['cosine_sim', 'scaled_dot'] #[False]
  pos_enc_dims = ['gdc'] #[16,64,128,256]

  for i, ds in enumerate(datas):
    for idx_i, idx in enumerate(indexes[i]):
      for pos_enc_dim in pos_enc_dims:
        opt['pos_enc_type'] = 'GDC' #pos_enc_dim #'DW' + str(pos_enc_dim)
        opt['rewiring'] = 'gdc'  #pos_enc_dim #'DW' + str(pos_enc_dim)

        print(f"Running Best Params for {ds}")
        opt["dataset"] = ds
        opt["folder"] = folders[i]
        opt["name"] = names[i]
        opt["index"] = indexes[i][idx_i]
        run_best_params(opt)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--folder', type=str, default=None, help='experiment folder to read')
  parser.add_argument('--index', type=int, default=0, help='index to take from experiment folder')
  parser.add_argument('--metric', type=str, default='accuracy', help='metric to sort the hyperparameter tuning runs on')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilise ODE learning')
  parser.add_argument('--run_with_KNN', action='store_true',
                      help='run parameters discovered without KNN, but now using KNN')
  parser.add_argument('--change_att_sim_type', action='store_true',
                      help='run parameters discovered with attention different attention similarity ')
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  # #todo if we add new pos_enc_types this will break
  # parser.add_argument('--pos_enc_type', type=str, default="GDC", help='positional encoder (default: GDC)')
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")
  parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                      help="use the adjoint ODE method to reduce memory footprint")
  parser.add_argument("--max_nfe", type=int, default=5000, help="Maximum number of function evaluations allowed.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")

  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

  args = parser.parse_args()

  opt = vars(args)
  # run_best_params(opt)
  # mainLoop(opt)
  # KNN_abalation(opt)
  # KNN_abalation_grid(opt)
  # run_top5withES(opt)
  # run_top5(opt)
  # edge_sampling_ablation(opt)
  # edge_sampling_online_ablation(opt)
  embeddings_ablation(opt)