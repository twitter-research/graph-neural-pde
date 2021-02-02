import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import pytorch_lightning as pl
from torch_geometric.nn.conv.spline_conv import SplineConv
# from torchdyn.models import NeuralDE
import torchdiffeq

# from torchdyn._internals import compat_check

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.TargetIndegree())
data = dataset[0]

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:data.num_nodes - 1000] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - 500:] = 1

defaults = {'type': 'classic', 'controlled': False, 'augment': False,  # model
            'backprop_style': 'autograd', 'cost': None,  # training
            's_span': torch.linspace(0, 1, 2), 'method': 'rk4', 'atol': 1e-3, 'rtol': 1e-4,  # method params
            'return_traj': False}


class NeuralDE(pl.LightningModule):
  """General Neural DE template

  :param func: function parametrizing the vector field.
  :type func: nn.Module
  :param settings: specifies parameters of the Neural DE.
  :type settings: dict
  """

  def __init__(self, func: nn.Module, opt: dict, device):
    super().__init__()
    # defaults.update(settings)
    # compat_check(defaults)
    self.opt = opt
    self.defunc, self.defunc.func_type = func, 'classic'
    self.defunc.controlled = False
    self.s_span = torch.tensor([0, opt['time']]).to(self.device)
    self.return_traj = False

    # check if integral
    # flag = (self.opt['backprop_style'] == 'integral_adjoint')
    self.set_tol()
    # self.adjoint = Adjoint(flag)

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def forward(self, x: torch.Tensor):
    return self._odesolve(x)

  def _odesolve(self, x: torch.Tensor):
    # TO DO: implement adaptive_depth check, insert here

    # assign control input and augment if necessary
    if self.defunc.controlled: self.defunc.u = x
    self.s_span = self.s_span.to(x)

    switcher = {
      'autograd': self._autograd,
      'integral_autograd': self._integral_autograd,
      'adjoint': self._adjoint,
      'integral_adjoint': self._integral_adjoint
    }
    # odeint = switcher.get(self.opt['backprop_style'])
    if opt['adjoint']:
      odeint = self._adjoint
    else:
      odeint = self._autograd
    # sol = odeint(x) if self.opt['return_traj'] else odeint(x)[-1]
    sol = odeint(x)[-1]
    return sol

  def trajectory(self, x: torch.Tensor, s_span: torch.Tensor):
    """Returns a data-flow trajectory at `s_span` points

    :param x: input data
    :type x: torch.Tensor
    :param s_span: collections of points to evaluate the function at e.g torch.linspace(0, 1, 100) for a 100 point trajectory
                   between 0 and 1
    :type s_span: torch.Tensor
    """
    if self.defunc.controlled: self.defunc.u = x
    sol = torchdiffeq.odeint(self.defunc, x, s_span,
                             rtol=self.rtol, atol=self.atol, method=self.opt['method'])
    return sol

  def backward_trajectory(self, x: torch.Tensor, s_span: torch.Tensor):
    assert self.adjoint, 'Propagating backward dynamics only possible with Adjoint systems'
    # register hook
    if self.defunc.controlled: self.defunc.u = x
    # set new s_span
    self.adjoint.s_span = s_span;
    x = x.requires_grad_(True)
    sol = self(x)
    sol.sum().backward()
    return sol.grad

  def _autograd(self, x):
    return torchdiffeq.odeint(self.defunc, x, self.s_span, rtol=self.rtol, atol=self.atol, method=self.opt['method'])

  def _adjoint(self, x):
    return torchdiffeq.odeint_adjoint(self.defunc, x, self.s_span, rtol=self.rtol, atol=self.atol,
                                      method=self.opt['method'])

  def _integral_adjoint(self, x):
    assert self.opt['cost'], 'Cost nn.Module needs to be specified for integral adjoint'
    return self.adjoint(self.defunc, x, self.s_span, cost=self.opt['cost'],
                        rtol=self.rtol, atol=self.atol, method=self.opt['method'])

  def _integral_autograd(self, x):
    assert self.opt['cost'], 'Cost nn.Module needs to be specified for integral adjoint'
    e0 = 0. * torch.ones(1).to(x.device)
    e0 = e0.repeat(x.shape[0]).unsqueeze(1)
    x = torch.cat([x, e0], 1)
    return torchdiffeq.odeint(self._integral_autograd_defunc, x, self.s_span,
                              rtol=self.rtol, atol=self.atol, method=self.opt['method'])

  def _integral_autograd_defunc(self, s, x):
    x = x[:, :-1]
    dxds = self.defunc(s, x)
    deds = self.settings['cost'](s, x).repeat(x.shape[0]).unsqueeze(1)
    return torch.cat([dxds, deds], 1)

  def __repr__(self):
    npar = sum([p.numel() for p in self.defunc.parameters()])
    return f"Neural DE\tType: {self.opt['type']}\tControlled: {self.opt['controlled']}\
        \nmethod: {self.opt['method']}\tIntegration interval: {self.opt['s_span'][0]} to {self.opt['s_span'][-1]}\
        \nCost: {self.opt['cost']}\tReturning trajectory: {self.opt['return_traj']}\
        \nTolerances: relative {self.opt['rtol']} absolute {self.opt['atol']}\
        \nFunction parametrizing vec. field:\n {self.defunc}\
        \n# parameters {npar}"


class GCNLayer(torch.nn.Module):
  def __init__(self, input_size, output_size, data, device):
    super(GCNLayer, self).__init__()

    if input_size != output_size:
      raise AttributeError('input size must equal output size')
    self.edge_index = data.edge_index.to(device)
    self.edge_attr = data.edge_attr.to(device)
    self.conv1 = SplineConv(input_size, output_size, dim=1, kernel_size=2).to(device)
    self.conv2 = SplineConv(input_size, output_size, dim=1, kernel_size=2).to(device)

  def forward(self, t, x):   # the t param is needed by the ODE solver.
    x = self.conv1(x, self.edge_index, self.edge_attr)
    x = self.conv2(x, self.edge_index, self.edge_attr)
    return x


class GDE(torch.nn.Module):
  def __init__(self, opt, data, device):
    super(GDE, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = data.edge_index.to(device)
    self.edge_attr = data.edge_attr.to(device)
    self.func = GCNLayer(input_size=opt['hidden_dim'], output_size=opt['hidden_dim'], data=data, device=device)

    self.conv1 = SplineConv(dataset.num_features, opt['hidden_dim'], dim=1, kernel_size=2).to(device)
    self.neuralDE = NeuralDE(self.func, opt, device).to(device)
    self.conv2 = SplineConv(opt['hidden_dim'], dataset.num_classes, dim=1, kernel_size=2).to(device)

  def forward(self, x):
    x = F.tanh(self.conv1(x, self.edge_index, self.edge_attr))
    x = F.dropout(x, p=self.opt['dropout'], training=self.training)
    x = self.neuralDE(x)
    x = F.tanh(self.conv2(x, self.edge_index, self.edge_attr))

    return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = dict(method='rk4', time=3, tol_scale=10, tol_scale_adjoint=10, hidden_dim=64, adjoint=False, dropout=0.5)
model, data = GDE(opt, data, device).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)


def train():
  model.train()
  optimizer.zero_grad()
  F.nll_loss(model(data.x)[data.train_mask], data.y[data.train_mask]).backward()
  optimizer.step()


def test():
  model.eval()
  logits, accs = model(data.x), []
  for _, mask in data('train_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs


for epoch in range(1, 20):
  train()
  log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
  print(log.format(epoch, *test()))
