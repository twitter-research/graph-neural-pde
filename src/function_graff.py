# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import shutil
import torch
from torch import nn
from torch.nn.init import uniform, xavier_uniform_
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import degree, softmax, homophily
from torch_sparse import coalesce, transpose
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from torch_scatter import scatter_mean
from torch.nn import Parameter, Softmax, Softplus
from torch.distributions import Categorical
from utils import MaxNFEException, sigmoid_deriv, tanh_deriv, squareplus_deriv
from base_classes import ODEFunc

class ODEFuncGraff(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncGraff, self).__init__(opt, data, device)
    self.in_features = in_features
    self.out_features = out_features
    self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.n_nodes = data.x.shape[0]
    self.deg_inv_sqrt = self.get_deg_inv_sqrt(data).to(device) #sending this to device because at initi, data is not yet sent to device
    self.deg_inv = self.deg_inv_sqrt * self.deg_inv_sqrt
    self.data = data

    self.num_timesteps = 1
    self.time_dep_w = self.opt['time_dep_w']
    self.time_dep_struct_w = self.opt['time_dep_struct_w']
    if self.time_dep_w or self.time_dep_struct_w:
      self.num_timesteps = math.ceil(self.opt['time']/self.opt['step_size'])

    #batch norm
    if self.opt['conv_batch_norm'] == "shared":
      self.batchnorm_h = nn.BatchNorm1d(in_features)
    elif self.opt['conv_batch_norm'] == "layerwise":
      nts = math.ceil(self.opt['time'] / self.opt['step_size'])
      self.batchnorms = [nn.BatchNorm1d(in_features).to(device) for _ in range(nts)]

    #init Omega
    if self.opt['omega_style'] == 'diag':
      if self.time_dep_w:
        self.om_W = Parameter(torch.Tensor(self.num_timesteps, in_features))
      else:
        self.om_W = Parameter(torch.Tensor(in_features))
    elif self.opt['omega_style'] == 'zero':
      self.om_W = torch.zeros((in_features,in_features), device=device)

    #init W
    if self.opt['w_style'] in ['sum', 'prod', 'neg_prod']:
      self.W_W = Parameter(torch.Tensor(in_features, in_features))
    elif self.opt['w_style'] == 'diag':
      if self.opt['w_diag_init'] == 'linear':
        d = in_features
        if self.time_dep_w or self.time_dep_struct_w:
          # This stores just the time dependent diagonals
          d_range = torch.tensor([list(range(d)) for _ in range(self.num_timesteps)], device=self.device)
          self.W_D = Parameter(
            self.opt['w_diag_init_q'] * d_range / (d-1) + self.opt['w_diag_init_r'],
            requires_grad=opt['w_param_free'])
          if self.time_dep_struct_w:
            self.brt = Parameter(
              -2. * torch.rand((self.num_timesteps, d), device=self.device) + 1,
              requires_grad=True
            )
            self.crt = Parameter(
              -2. * torch.rand((self.num_timesteps, d), device=self.device) + 1,
              requires_grad=True
            )
            self.drt = Parameter(
              -2. * torch.rand((self.num_timesteps, d), device=self.device) + 1,
              requires_grad=True
            )
        else:
          d_range = torch.tensor(list(range(d)), device=self.device)
          self.W_D = Parameter(self.opt['w_diag_init_q'] * d_range / (d-1) + self.opt['w_diag_init_r'], requires_grad=opt['w_param_free'])
      else:
        if self.time_dep_w or self.time_dep_struct_w:
          self.W_D = Parameter(torch.ones(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
          if self.time_dep_struct_w:
            self.brt = Parameter(
              -2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1,
              requires_grad=True
            )
            self.crt = Parameter(
              -2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1,
              requires_grad=True
            )
            self.drt = Parameter(
              -2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1,
              requires_grad=True
            )
        else:
          self.W_D = Parameter(torch.ones(in_features), requires_grad=opt['w_param_free'])
    elif self.opt['w_style'] == 'diag_dom':
      self.W_W = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['w_param_free'])
      self.t_a = Parameter(torch.Tensor(in_features), requires_grad=opt['w_param_free'])
      self.r_a = Parameter(torch.Tensor(in_features), requires_grad=opt['w_param_free'])
      if self.time_dep_w:
        self.t_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
        self.r_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
      if self.time_dep_struct_w:
        self.at = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
        self.bt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
        self.gt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)

    self.reset_parameters()

  def reset_parameters(self):
    # Omega
    if self.opt['omega_style'] == 'diag':
      uniform(self.om_W, a=-1, b=1)
    # W's
    if self.opt['w_style'] in ['sum','prod','neg_prod']:
      glorot(self.W_W)
    elif self.opt['w_style'] == 'diag':
      if self.opt['w_diag_init'] == 'uniform':
        uniform(self.W_D, a=-1, b=1)
        if self.time_dep_struct_w:
          uniform(self.brt, a=-1, b=1)
          uniform(self.crt, a=-1, b=1)
          uniform(self.drt, a=-1, b=1)
      elif self.opt['w_diag_init'] == 'identity':
        ones(self.W_D)
      elif self.opt['w_diag_init'] == 'linear':
        pass #done in init
    elif self.opt['w_style'] == 'diag_dom':
      if self.time_dep_struct_w:
        uniform(self.at, a=-1, b=1)
        uniform(self.bt, a=-1, b=1)
        uniform(self.gt, a=-1, b=1)
      if self.opt['w_diag_init'] == 'uniform':
        glorot(self.W_W)
        uniform(self.t_a, a=-1, b=1)
        uniform(self.r_a, a=-1, b=1)
      elif self.opt['w_diag_init'] == 'identity':
        zeros(self.W_W)
        constant(self.t_a, fill_value=1)
        constant(self.r_a, fill_value=1)
      elif self.opt['w_diag_init'] == 'linear':
        glorot(self.W_W)
        constant(self.t_a, fill_value=self.opt['w_diag_init_q'])
        constant(self.r_a, fill_value=self.opt['w_diag_init_r'])

  def get_deg_inv_sqrt(self, data):
    index_tensor = data.edge_index[0]
    deg = degree(index_tensor, self.n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    return deg_inv_sqrt

  def get_src_dst(self, x):
    """
    Get the values of a dense n-by-d matrix
    @param x:
    @return:
    """
    src = x[self.edge_index[0, :]]
    dst = x[self.edge_index[1, :]]
    return src, dst

  def set_Omega(self, T=None):
    if self.opt['omega_style'] == 'diag':
      if self.opt['omega_diag'] == 'free':
        if self.time_dep_w:
          if T is None:
            T = 0
          Omega = torch.diag(self.om_W[T])
        else:
          Omega = torch.diag(self.om_W)
      elif self.opt['omega_diag'] == 'const':
        Omega = torch.diag(self.opt['omega_diag_val'] * torch.ones(self.in_features, device=self.device))
    elif self.opt['omega_style'] == 'zero':
      Omega = torch.zeros((self.in_features,self.in_features), device=self.device)
    return Omega

  def set_W(self, T=None):
    if T is None:
      T = 0
    "note every W is made symetric before returning here"
    if self.opt['w_style'] in ['prod']:
      return self.W_W @ self.W_W.t()
    elif self.opt['w_style'] in ['neg_prod']:
      return -self.W_W @ self.W_W.t()
    elif self.opt['w_style'] in ['sum']:
      return (self.W_W + self.W_W.t()) / 2
    elif self.opt['w_style'] == 'diag':
      if self.time_dep_w:
        if T is None:
          T = 0
        return torch.diag(self.W_D[T])
      elif self.time_dep_struct_w:
        if T is None:
          T = 0
        W = self.W_D[T]
        alpha = torch.diag(torch.exp(self.brt[T] * T + self.brt[T]))
        beta = torch.diag(torch.exp(-self.brt[T] * T - self.crt[T]) + self.drt[T])
        Wplus = torch.diag(F.relu(W))
        Wneg = torch.diag(-1. * F.relu(-W))
        return alpha @ Wplus - beta @ Wneg
      else:
        return torch.diag(self.W_D)
    elif self.opt['w_style'] == 'diag_dom':
      W_temp = torch.cat([self.W_W, torch.zeros((self.in_features, 1), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      W = (W+W.T) / 2
      if self.time_dep_w:
        W_sum = self.t_a[T] * torch.abs(W).sum(dim=1) + self.r_a[T]
      elif self.time_dep_struct_w:
        W_sum = W + self.at[T] * F.tanh(self.bt[T] * T + self.gt[T]) * torch.eye(n=W.shape[0], m=W.shape[1], device=self.device)
      else:
         W_sum = self.t_a * torch.abs(W).sum(dim=1) + self.r_a
      Ws = W + torch.diag(W_sum)
      return Ws
    elif self.opt['w_style'] == 'identity':
      return torch.eye(self.in_features, device=self.device)


  def forward(self, t, x):
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    if self.time_dep_w:
      T = int(t / self.opt['step_size'])
      self.W = self.set_gnlWS(T)
      self.Omega = self.set_gnlOmega(T)
    else:
      T = 0

    src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)
    attention = torch.ones(src_deginvsqrt.shape, device=self.device)
    P = attention * src_deginvsqrt * dst_deginvsqrt
    xW = x @ self.W
    f = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], xW)
    f = f - x @ self.Omega

    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    if self.opt['conv_batch_norm'] == "shared":
      f = self.batchnorm_h(f)
    elif self.opt['conv_batch_norm'] == "layerwise":
      f = self.batchnorms[T](f)

    #non-linearity
    if self.opt['pointwise_nonlin']:
      f = torch.relu(f)

    return f

def __repr__(self):
  return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'