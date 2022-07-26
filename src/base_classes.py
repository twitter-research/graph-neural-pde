import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import uniform, xavier_uniform_
from torch_geometric.nn.conv import MessagePassing
from utils import Meter
from regularized_ODE_function import RegularizedODEfunc
import regularized_ODE_function as reg_lib
import six
import wandb

REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs


class ODEblock(nn.Module):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t
    
    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    
    self.nreg = len(regularization_fns)
    self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    self.reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self, opt, data, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1)) #todo are these even used
    self.beta_sc = nn.Parameter(torch.ones(1)) #todo are these even used

  def __repr__(self):
    return self.__class__.__name__


class BaseGNN(MessagePassing):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(BaseGNN, self).__init__()
    self.opt = opt
    self.T = opt['time']
    self.num_classes = dataset.num_classes
    self.num_features = dataset.data.num_features
    self.num_nodes = dataset.data.num_nodes
    self.device = device
    self.fm = Meter()
    self.bm = Meter()

    if opt['beltrami']:
      self.mx = nn.Linear(self.num_features, opt['feat_hidden_dim'])
      self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])

      if opt['wandb']:
        wandb.config.update({'hidden_dim': opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']}, allow_val_change=True)  # required when update hidden_dim in beltrami
      else:
        opt['hidden_dim'] = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']
    else:
      self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])

    if self.opt['use_mlp']:
      self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    if opt['use_labels']:
      # todo - fastest way to propagate this everywhere, but error prone - refactor later
      opt['hidden_dim'] = opt['hidden_dim'] + dataset.num_classes
    else:
      self.hidden_dim = opt['hidden_dim']
    if opt['fc_out']:
      self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])

    if opt['m2_mlp']:
      self.m2 = M2_MLP(opt['hidden_dim'], dataset.num_classes, opt)
    else:
      self.m2 = nn.Linear(opt['hidden_dim'], dataset.num_classes)

    if self.opt['path_dep_norm'] == 'z_cat_normed_z':
      self.m2_concat = nn.Linear(opt['hidden_dim'], dataset.num_classes)
      self.alpha_z = nn.Parameter(torch.ones(1)/2) #todo are these even used

    out_dim = 2 * opt['hidden_dim'] if self.opt['path_dep_norm'] == 'z_cat_normed_z' else opt['hidden_dim']
    time_points = math.ceil(opt['time']/opt['step_size'])
    if self.opt['m3_path_dep'] == 'feature_jk':
      self.m3 = nn.Linear((time_points + 1) * out_dim, self.num_classes)
    elif self.opt['m3_path_dep'] == 'label_jk':
      self.m3 = nn.Linear((time_points + 1) * self.num_classes, self.num_classes)
    elif self.opt['m3_path_dep'] == 'label_att':
      # self.label_atts = nn.Parameter(torch.Tensor(time_points + 1))
      self.label_atts = nn.Parameter(torch.ones(time_points + 1))
    elif self.opt['m3_path_dep'] == 'train_centers':
      self.m3 = nn.Linear((time_points + 1) * self.num_classes, self.num_classes)


    if self.opt['batch_norm']:
      self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
      self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])

    self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

  def getNFE(self):
    return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()
    if self.opt['m3_path_dep'] in ['feature_jk', 'label_jk']:
      self.m3.reset_parameters()
    elif self.opt['m3_path_dep'] == 'label_att':
      pass # xavier_uniform_(self.label_atts)
    if self.opt['path_dep_norm'] == 'z_cat_normed_z':
      pass

  def __repr__(self):
    return self.__class__.__name__

class M2_MLP(nn.Module):
  def __init__(self, out_dim, num_classes, opt):
    super().__init__()
    self.opt = opt
    self.m21 = nn.Linear(out_dim, out_dim)
    self.m22 = nn.Linear(out_dim, num_classes)

  def forward(self, x):
    x = F.dropout(x, self.opt['dropout'], training=self.training)
    x = F.dropout(x + self.m21(torch.tanh(x)), self.opt['dropout'], training=self.training)  # tanh not relu to keep sign, with skip connection
    x = F.dropout(self.m22(torch.tanh(x)), self.opt['dropout'], training=self.training)

    return x