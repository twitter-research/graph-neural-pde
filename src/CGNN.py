"""
A reimplementation of Continous Graph Neural Networks https://github.com/DeepGraphLearning/ContinuousGNN
@misc{xhonneux2019continuous,
    title={Continuous Graph Neural Networks},
    author={Louis-Pascal A. C. Xhonneux and Meng Qu and Jian Tang},
    year={2019},
    eprint={1912.00967},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
to use pytorch geometric data pipelines
"""


import time
import os
import torch
import argparse
from torch import nn
import torch.nn.functional as F
from data import get_dataset
# Whether use adjoint method or not.
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import numpy as np
from utils import Meter
from ray import tune
from functools import partial
from ray.tune import CLIReporter
from utils import get_sem, mean_confidence_interval
from utils import gcn_norm_fill_val
from data import set_train_val_test_split

adjoint = False
if adjoint:
  from torchdiffeq import odeint_adjoint as odeint
else:
  from torchdiffeq import odeint

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat, pos_encoding), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, adj):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.adj = adj
    self.x0 = None
    self.nfe = 0
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['alpha']
    self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))

    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)

  def forward(self, t, x):
    self.nfe += 1
    alph = torch.sigmoid(self.alpha_train).unsqueeze(dim=1)
    ax = torch.spmm(self.adj, x)
    f = alph * 0.5 * (ax - x) + self.x0
    return f


class ODEblock(nn.Module):
  def __init__(self, odefunc, t=torch.tensor([0, 1])):
    super(ODEblock, self).__init__()
    self.t = t
    self.odefunc = odefunc
    self.nfe = 0

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()

  def forward(self, x):

    t = self.t.type_as(x)
    z = odeint(self.odefunc, x, t)[1]
    self.nfe += self.odefunc.nfe
    self.odefunc.nfe = 0
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


# Define the GNN model.
class CGNN(nn.Module):
  def __init__(self, opt, adj, time, device):
    super(CGNN, self).__init__()
    self.opt = opt
    self.adj = adj
    self.T = time
    self.fm = Meter()
    self.bm = Meter()
    self.device = device
    self.m1 = nn.Linear(opt['num_feature'], opt['hidden_dim'])

    self.odeblock = ODEblock(ODEFunc(2 * opt['hidden_dim'], 2 * opt['hidden_dim'], opt, adj),
                             t=torch.tensor([0, self.T]))

    self.m2 = nn.Linear(opt['hidden_dim'], opt['num_class'])

  def getNFE(self):
    return self.odeblock.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def forward(self, x, pos_enc=None):
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

    # Solve the initial value problem of the ODE.
    c_aux = torch.zeros(x.shape).to(self.device)
    x = torch.cat([x, c_aux], dim=1)
    self.odeblock.set_x0(x)

    z = self.odeblock(x)
    z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z


def get_cora_opt(opt):
  opt['dataset'] = 'Cora'
  #opt['data'] = 'Planetoid'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['decay'] = 5e-4

  if opt["num_splits"] == 0:
    opt['lr'] = 0.0047
    opt['self_loop_weight'] = 0.555
    opt['alpha'] = 0.918
    opt['time'] = 12.1
  else:
    opt['lr'] = 0.00147
    opt['self_loop_weight'] = 0.595
    opt['alpha'] = 0.885
    opt['time'] = 23.9

  opt['epoch'] = 100
  opt['num_feature'] = 1433
  opt['num_class'] = 7
  opt['num_nodes'] = 2708
  opt['augment'] = True
  opt['attention_dropout'] = 0
  opt['adjoint'] = False
  opt['ode'] = 'ode'
  return opt


def get_citeseer_opt(opt):
  opt['dataset'] = 'Citeseer'
  #opt['data'] = 'Planetoid'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.00548
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.758
  opt['alpha'] = 0.869

  if opt["num_splits"] == 0:
    opt['lr'] = 0.00548
    opt['self_loop_weight'] = 0.758
    opt['alpha'] = 0.869
    opt['time'] = 19.1
  else:
    opt['lr'] = 0.00298
    opt['self_loop_weight'] = 0.459
    opt['alpha'] = 0.936
    opt['time'] = 17.1

  opt['epoch'] = 100
  opt['num_feature'] = 3703
  opt['num_class'] = 6
  opt['num_nodes'] = 3327
  opt['augment'] = True
  opt['attention_dropout'] = 0
  opt['adjoint'] = False
  opt['ode'] = 'ode'
  return opt


def get_pubmed_opt(opt):
  opt['dataset'] = 'Pubmed'
  #opt['data'] = 'Planetoid'
  
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'adam'
  opt['decay'] = 5e-4

  if opt["num_splits"] == 0:
    opt['lr'] = 0.0054
    opt['self_loop_weight'] = 0.644
    opt['alpha'] = 0.96
    opt['time'] = 16.2
  else:
    opt['lr'] = 0.00551
    opt['self_loop_weight'] = 0.752
    opt['alpha'] = 0.947
    opt['time'] = 22.0

  opt['epoch'] = 100
  opt['num_feature'] = 500
  opt['num_class'] = 3
  opt['num_nodes'] = 19717
  opt['augment'] = True
  opt['attention_dropout'] = 0
  opt['adjoint'] = False
  opt['ode'] = 'ode'
  return opt


def coo2tensor(coo, device):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, device):
  edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, data.edge_attr, opt['self_loop_weight'], data.num_nodes,
                                              dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo, device)


def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  out = model(data.x)
  lf = torch.nn.CrossEntropyLoss()
  loss = lf(out[data.train_mask], data.y[data.train_mask]) 
  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


def main(opt):
  try:
    if opt['use_cora_defaults']:
      opt = get_cora_opt(opt)
  except KeyError:
    pass  # not always present when called as lib

  dataset = get_dataset(opt, '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  adj = get_sym_adj(dataset.data, opt, device)
  model, data = CGNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
  print(opt)

  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = best_epoch = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))

  return train_acc, best_val_acc, test_acc


def train_ray(opt, checkpoint_dir=None, data_dir='../data', opt_val=False):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = get_dataset(opt, data_dir, False)
  adj = get_sym_adj(dataset.data, opt, device)
  model, data = CGNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  model, data = model.to(device), dataset.data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
  # should be restored.
  if checkpoint_dir:
    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt['epoch']):
    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save(
        (model.state_dict(), optimizer.state_dict()), path)
    if opt_val:
      tune.report(loss=loss, accuracy=val_acc)
    else:
      tune.report(loss=loss, accuracy=tmp_test_acc)


def train_ray_icml(opt, checkpoint_dir=None, data_dir="../data", opt_val=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, False)

  if opt["num_splits"] > 0:
    dataset.data = set_train_val_test_split(
      23 * np.random.randint(0, opt["num_splits"]), 
      dataset.data, 
      num_development = 5000 if opt["dataset"] == "CoauthorCS" else 1500)

  adj = get_sym_adj(dataset.data, opt, device)
  model, data = CGNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
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

  for epoch in range(1, opt["epoch"]):
    loss = train(model, optimizer, data)
    # need next line as it sets the attributes in the solver
    _, val_acc_int, tmp_test_acc_int = test(model, data)
    
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)
    if opt_val:
      tune.report(loss=loss, accuracy=val_acc_int)
    else:
      tune.report(loss=loss, accuracy=tmp_test_acc_int)


def run_best_params(opt):
  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "training_iteration"])
  if opt['dataset'] == 'Cora':
    best_params = get_cora_opt(opt)
  elif opt['dataset'] == 'Citeseer':
    best_params = get_citeseer_opt(opt)
  elif opt['dataset'] == 'Pubmed':
    best_params = get_pubmed_opt(opt)
  else:
    best_params = opt
  print(opt)
  result = tune.run(
    partial(train_ray_icml, data_dir=data_dir),
    name=opt['name'],
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    config=best_params,
    num_samples=opt['num_splits'],
    scheduler=None,
    max_failures=3,
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric="accuracy", mode="max").sort_values('accuracy',
                                                                   ascending=False)
  print(df['accuracy'])

  test_accs = df['accuracy'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--save', type=str, default='/')
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
  parser.add_argument('--use_gold', type=int, default=1,
                      help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
  parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--draw', type=str, default='max',
                      help='Method for drawing object labels, max for max-pooling, smp for sampling.')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--ode', type=str, default='ode', help="set ode block. Either 'ode', 'att', 'sde'")
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--rtol', type=float, default=1e-5,
                      help='relative error tolerance in adaptive step size solvers')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--reps', type=int, default=30, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default='ray_test')
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")

  args = parser.parse_args()
  opt = vars(args)

  run_best_params(opt)
