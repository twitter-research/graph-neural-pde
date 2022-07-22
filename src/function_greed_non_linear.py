"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""
import math
import os
import shutil
import torch
from torch import nn
from torch.nn.init import uniform, xavier_uniform_
import torch.nn.functional as F
import numpy as np
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import degree, softmax, homophily, contains_self_loops, to_dense_adj
from torch_sparse import coalesce, transpose
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from torch_scatter import scatter_mean
from torch.nn import Parameter, Softmax, Softplus
from torch.distributions import Categorical
import wandb
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix

from function_greed import ODEFuncGreed
from utils import MaxNFEException, sym_row_col, sym_row_col_att, sym_row_col_att_measure, gram_schmidt, sym_row_col_att_relaxed, sigmoid_deriv, tanh_deriv, squareplus_deriv
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer
from function_transformer_attention_greed import SpGraphTransAttentionLayer_greed
from greed_reporting_fcts import set_reporting_attributes, set_folders_pdfs, generate_stats, append_stats, stack_stats

class ODEFuncGreedNonLin(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedNonLin, self).__init__(in_features, out_features, opt, data, device, bias=False)
    self.data = data

    if self.opt['wandb_track_grad_flow']:
      set_reporting_attributes(self, data, opt)
      set_folders_pdfs(self, opt)

    self.num_timesteps = 1
    self.time_dep_w = self.opt['time_dep_w']
    self.time_dep_struct_w = self.opt['time_dep_struct_w']
    if self.time_dep_w or self.time_dep_struct_w:
      self.num_timesteps = math.ceil(self.opt['time']/self.opt['step_size'])

    self.paths = []
    self.epoch = 0
    self.wandb_step = 0
    self.prev_grad = None

    #Note - Omega params are used differently for scaled_dot and general graph
    if self.opt['gnl_omega'] == 'sum':
      self.om_W = Parameter(torch.Tensor(in_features, in_features))
      self.om_W_eps = 0
    elif self.opt['gnl_omega'] == 'product':
      self.om_W = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
    elif self.opt['gnl_omega'] == 'attr_rep':
      self.om_W_attr = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
      self.om_W_rep = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
      self.om_W_eps = Parameter(torch.Tensor([0.85]))
      self.om_W_nu = torch.Tensor([0.1], device=self.device)
    elif self.opt['gnl_omega'] == 'diag':
      if self.time_dep_w:
        self.om_W = Parameter(torch.Tensor(self.num_timesteps, in_features))
      else:
        self.om_W = Parameter(torch.Tensor(in_features))
      self.om_W_eps = 0
    elif self.opt['gnl_omega'] == 'Omega_W_eig':
      self.om_W = Parameter(torch.Tensor(in_features))
      self.om_W_eps = 0
    elif self.opt['gnl_omega'] == 'zero':
      self.om_W = torch.zeros((in_features,in_features), device=device)
      self.om_W_eps = 0
    elif self.opt['gnl_omega'] == 'Omega_eq_W':
      self.om_W_eps = 0

    if opt['gnl_measure'] in ['deg_poly', 'deg_poly_exp']:
      self.m_alpha = Parameter(torch.Tensor([1.]))
      self.m_beta = Parameter(torch.Tensor([1.]))
      self.m_gamma = Parameter(torch.Tensor([0.]))
    elif opt['gnl_measure'] in ['nodewise', 'nodewise_exp']:
      self.measure = Parameter(torch.Tensor(self.n_nodes))
    elif opt['gnl_measure'] == 'ones':
      pass

    #'gnl_style' in 'scaled_dot' / 'softmax_attention' / 'general_graph'
    if self.opt['gnl_style'] == 'softmax_attention' or self.opt['gnl_attention']:
      self.multihead_att_layer = SpGraphTransAttentionLayer_greed(in_features, out_features, opt, device, edge_weights=self.edge_weight).to(device)

    if self.opt['gnl_style'] == 'general_graph':
      # gnl_omega -> "gnl_W"
      if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod']:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))
        if self.opt['two_hops']:
          self.W_W_tilde = Parameter(torch.Tensor(in_features, in_features))
      elif self.opt['gnl_W_style'] in ['Z_diag']:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))
      elif self.opt['gnl_W_style'] == 'diag':
        if self.opt['gnl_W_diag_init'] == 'linear':
          d = in_features
          if self.time_dep_w or self.time_dep_struct_w:
            # This stores just the time dependent diagonals
            d_range = torch.tensor([list(range(d)) for _ in range(self.num_timesteps)], device=self.device)
            self.gnl_W_D = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (d-1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
            if self.time_dep_struct_w:
              self.brt = Parameter(-2. * torch.rand((self.num_timesteps, d), device=self.device) + 1, requires_grad=True)
              self.crt = Parameter(-2. * torch.rand((self.num_timesteps, d), device=self.device) + 1, requires_grad=True)
              self.drt = Parameter(-2. * torch.rand((self.num_timesteps, d), device=self.device) + 1, requires_grad=True)
          else:
            d_range = torch.tensor(list(range(d)), device=self.device)
            self.gnl_W_D = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (d-1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
          if self.opt['two_hops']:
            self.gnl_W_D_tilde = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (d-1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
        else:
          if self.time_dep_w or self.time_dep_struct_w:
            self.gnl_W_D = Parameter(torch.ones(self.num_timesteps, in_features), requires_grad=opt['gnl_W_param_free'])
            if self.time_dep_struct_w:
              self.brt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
              self.crt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
              self.drt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
          else:
            self.gnl_W_D = Parameter(torch.ones(in_features), requires_grad=opt['gnl_W_param_free'])
          if self.opt['two_hops']:
            self.gnl_W_D_tilde = Parameter(torch.ones(in_features), requires_grad=opt['gnl_W_param_free'])
      elif self.opt['gnl_W_style'] == 'diag_dom':
        self.W_W = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['gnl_W_param_free'])
        self.t_a = Parameter(torch.Tensor(in_features), requires_grad=opt['gnl_W_param_free'])
        self.r_a = Parameter(torch.Tensor(in_features), requires_grad=opt['gnl_W_param_free'])
        if self.time_dep_w:
          self.t_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['gnl_W_param_free'])
          self.r_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['gnl_W_param_free'])
        if self.time_dep_struct_w:
          self.at = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
          self.bt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
          self.gt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
        if self.opt['two_hops']:
          self.W_W_tilde = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['gnl_W_param_free'])
      elif self.opt['gnl_W_style'] == 'k_diag_pc':
        k_num = int(self.opt['k_diag_pc'] * in_features)
        if k_num % 2 == 0:
          k_num += 1
        k_num = min(k_num, in_features)
        self.gnl_W_diags = Parameter(torch.Tensor(in_features, k_num))
        if self.opt['two_hops']:
          self.gnl_W_diags_tilde = Parameter(torch.Tensor(in_features, k_num))
      elif self.opt['gnl_W_style'] == 'k_block':
        assert opt['k_blocks'] * opt['block_size'] <= in_features, 'blocks exceeded hidden dim'
        self.gnl_W_blocks = Parameter(torch.Tensor(opt['k_blocks'] * opt['block_size'], opt['block_size']))
        self.gnl_W_D = Parameter(torch.Tensor(in_features - opt['k_blocks'] * opt['block_size']))
        if self.opt['two_hops']:
          self.gnl_W_blocks_tilde = Parameter(torch.Tensor(opt['k_blocks'] * opt['block_size'], opt['block_size']))
          self.gnl_W_D_tilde = Parameter(torch.Tensor(in_features - opt['k_blocks'] * opt['block_size']))
      elif self.opt['gnl_W_style'] == 'k_diag':
        assert opt['k_diags'] % 2 == 1 and opt['k_diags'] <= in_features, 'must have odd number of k diags'
        self.gnl_W_diags = Parameter(torch.Tensor(in_features, opt['k_diags'])) #or (2k-1) * n + k * (k - 1) if don't wrap around
        if self.opt['two_hops']:
          self.gnl_W_diags_tilde = Parameter(torch.Tensor(in_features, opt['k_diags']))
      elif self.opt['gnl_W_style'] in ['GS', 'GS_Z_diag']:
        self.gnl_W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
        if self.opt['two_hops']:
          self.gnl_W_U_tilde = Parameter(torch.Tensor(in_features, in_features))
          self.gnl_W_D_tilde = Parameter(torch.ones(in_features))          
      elif self.opt['gnl_W_style'] == 'cgnn':
        self.gnl_W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
        if self.opt['two_hops']:
          self.gnl_W_U_tilde = Parameter(torch.Tensor(in_features, in_features))
          self.gnl_W_D_tilde = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'feature':
        self.Om_phi = Parameter(torch.Tensor(in_features))
        self.W_psi = Parameter(torch.Tensor(in_features))
        if self.opt['two_hops']:
          self.Om_phi_tilde = Parameter(torch.Tensor(in_features))
          self.W_psi_tilde = Parameter(torch.Tensor(in_features))
      elif self.opt['gnl_W_style'] == 'positional':
        self.phi = nn.Linear(self.opt['pos_enc_hidden_dim'], self.in_features)
        self.psi = nn.Linear(self.opt['pos_enc_hidden_dim'], self.in_features)

    elif self.opt['gnl_style'] == 'att_rep_laps':
      if self.opt['gnl_W_style'] == 'att_rep_lap_block':
        half_in_features = int(in_features/2)
        self.L_W = Parameter(torch.Tensor(half_in_features, half_in_features - 1), requires_grad=opt['gnl_W_param_free'])
        self.L_t_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
        self.L_r_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
        self.R_W = Parameter(torch.Tensor(half_in_features, half_in_features - 1), requires_grad=opt['gnl_W_param_free'])
        self.R_t_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
        self.R_r_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
      elif self.opt['gnl_W_style'] == 'sum':
        self.W_W = Parameter(torch.Tensor(in_features, in_features))

    self.delta = Parameter(torch.Tensor([1.]))
    self.C = (data.y.max() + 1).item()  #hack!, num class for drift
    if opt['drift'] or opt['lie_trotter'] in ['gen_0','gen_1','gen_2']:
      self.drift_eps = Parameter(torch.Tensor([0.]))

    # self.attractors = {i: Parameter(torch.Tensor(opt['hidden_dim'])) for i in range(self.C)}
    # self.attractors = {i: Parameter(F.one_hot(torch.tensor([i], dtype=torch.long), num_classes=self.C).type(torch.float)) for i in range(self.C)}
    if self.opt['drift_space'] == 'feature':
      self.attractors = {}
      for i in range(self.C):
        z = torch.zeros(opt['hidden_dim'])
        z[i] = 1.
        self.attractors[i] = Parameter(z)

    self.reset_nonlinG_parameters()


  def reset_nonlinG_parameters(self):
    # Omega
    if self.opt['gnl_omega'] == 'sum':
      glorot(self.om_W)
    elif self.opt['gnl_omega'] == 'product':
      glorot(self.om_W)
    elif self.opt['gnl_omega'] == 'attr_rep':
      glorot(self.om_W_attr)
      glorot(self.om_W_rep)
    elif self.opt['gnl_omega'] == 'diag':
      uniform(self.om_W, a=-1, b=1)
    elif self.opt['gnl_omega'] == 'Omega_W_eig':
      uniform(self.om_W, a=-1, b=1)

    # W's
    if self.opt['gnl_style'] == 'general_graph':
      if self.opt['gnl_W_style'] in ['sum','prod','neg_prod']:
        glorot(self.W_W)
        if self.opt['two_hops']:
          glorot(self.W_W_tilde)
      elif self.opt['gnl_W_style'] in ['Z_diag']:
        glorot(self.W_W)
      elif self.opt['gnl_W_style'] == 'diag':
        if self.opt['gnl_W_diag_init'] == 'uniform':
          uniform(self.gnl_W_D, a=-1, b=1)
          if self.opt['two_hops']:
            uniform(self.gnl_W_D_tilde, a=-1, b=1)
          if self.time_dep_struct_w:
            uniform(self.brt, a=-1, b=1)
            uniform(self.crt, a=-1, b=1)
            uniform(self.drt, a=-1, b=1)
        elif self.opt['gnl_W_diag_init'] == 'identity':
          ones(self.gnl_W_D)
          if self.opt['two_hops']:
            ones(self.gnl_W_D_tilde)
        elif self.opt['gnl_W_diag_init'] == 'linear':
          pass #done in init
      elif self.opt['gnl_W_style'] == 'diag_dom':
        if self.time_dep_struct_w:
          uniform(self.at, a=-1, b=1)
          uniform(self.bt, a=-1, b=1)
          uniform(self.gt, a=-1, b=1)
        if self.opt['gnl_W_diag_init'] == 'uniform':
          #todo regularise wrt hidden_dim as summing abs(W) off diags
          #todo initialise spectrum distribution proportional to graph homophily
          glorot(self.W_W)
          uniform(self.t_a, a=-1, b=1)
          uniform(self.r_a, a=-1, b=1)
          if self.opt['two_hops']:
            glorot(self.W_W_tilde)
            uniform(self.t_a_tilde, a=-1, b=1)
            uniform(self.r_a_tilde, a=-1, b=1)
        elif self.opt['gnl_W_diag_init'] == 'identity':
          zeros(self.W_W)
          constant(self.t_a, fill_value=1)
          constant(self.r_a, fill_value=1)
          if self.opt['two_hops']:
            zeros(self.W_W_tilde)
            constant(self.t_a_tilde, fill_value=1)
            constant(self.r_a_tilde, fill_value=1)
        elif self.opt['gnl_W_diag_init'] == 'linear':
          glorot(self.W_W)
          constant(self.t_a, fill_value=self.opt['gnl_W_diag_init_q'])
          constant(self.r_a, fill_value=self.opt['gnl_W_diag_init_r'])
          if self.opt['two_hops']:
            glorot(self.W_W_tilde)
            constant(self.t_a_tilde, fill_value=self.opt['gnl_W_diag_init_q'])
            constant(self.r_a_tilde, fill_value=self.opt['gnl_W_diag_init_r'])
      elif self.opt['gnl_W_style'] == 'k_block':
        glorot(self.gnl_W_blocks)
        uniform(self.gnl_W_D, a=-1, b=1)
        if self.opt['two_hops']:
          glorot(self.gnl_W_blocks_tilde)
          uniform(self.gnl_W_D_tilde, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'k_diag':
        uniform(self.gnl_W_diags, a=-1, b=1)
        if self.opt['two_hops']:
          uniform(self.gnl_W_diags_tilde, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'k_diag_pc':
        if self.opt['gnl_W_diag_init'] == 'uniform':
          uniform(self.gnl_W_diags, a=-1, b=1)
          if self.opt['two_hops']:
            uniform(self.gnl_W_diags_tilde, a=-1, b=1)
        elif self.opt['gnl_W_diag_init'] == 'identity':
          ones(self.gnl_W_diags)
          if self.opt['two_hops']:
            ones(self.gnl_W_diags_tilde)
      elif self.opt['gnl_W_style'] in ['GS', 'GS_Z_diag']:
        glorot(self.gnl_W_U)
        uniform(self.gnl_W_D, a=-1, b=1)
        if self.opt['two_hops']:
          glorot(self.gnl_W_U_tilde)
      elif self.opt['gnl_W_style'] == 'cgnn':
        glorot(self.gnl_W_U)
        if self.opt['two_hops']:
          glorot(self.gnl_W_U_tilde)
      elif self.opt['gnl_W_style'] == 'feature':
        glorot(self.Om_phi)
        glorot(self.W_psi)
        if self.opt['two_hops']:
          glorot(self.Om_phi_tilde)
          glorot(self.W_psi_tilde)
      elif self.opt['gnl_W_style'] == 'positional':
        pass #linear layer
    elif self.opt['gnl_style'] == 'att_rep_laps':
      if self.opt['gnl_W_style'] == 'att_rep_lap_block':
        glorot(self.L_W)
        constant(self.L_t_a, fill_value=self.opt['gnl_W_diag_init_q'])
        constant(self.L_r_a, fill_value=self.opt['gnl_W_diag_init_r'])
        glorot(self.R_W)
        constant(self.R_t_a, fill_value=self.opt['gnl_W_diag_init_q'])
        constant(self.R_r_a, fill_value=self.opt['gnl_W_diag_init_r'])
      elif self.opt['gnl_W_style'] == 'sum':
        glorot(self.W_W)


    if self.opt['gnl_measure'] in ['deg_poly', 'deg_poly_exp']:
      ones(self.m_alpha)
      ones(self.m_beta)
      ones(self.m_gamma)
    elif self.opt['gnl_measure'] in ['nodewise']:
      ones(self.measure)
    elif self.opt['gnl_measure'] in ['nodewise_exp']:
      zeros(self.measure)

    # for c in self.attractors.values():
    #   zeros(c)

  def set_scaled_dot_omega(self, T=None):
    if self.opt['gnl_omega'] == 'sum':
      Omega = self.om_W + self.om_W.T
    elif self.opt['gnl_omega'] == 'product':
      Omega = self.om_W @ self.om_W.T
    elif self.opt['gnl_omega'] == 'attr_rep':
      # Omega = self.om_W_nu * (1 - 2 * self.om_W_eps) - self.om_W_eps * self.om_W_attr @ self.om_W_attr.T + (1 - self.om_W_eps) * self.om_W_rep @ self.om_W_rep.T
      Omega = (1 - 2 * self.om_W_eps) * torch.eye(self.in_features, device=self.device) - self.om_W_eps * self.om_W_attr @ self.om_W_attr.T + (
                             1 - self.om_W_eps) * self.om_W_rep @ self.om_W_rep.T
    elif self.opt['gnl_omega'] == 'diag':
      if self.time_dep_w:
        Omega = torch.diag(self.om_W[T])
      else:
        Omega = torch.diag(self.om_W)
    # method for normalising Omega to control the eigen values
    if self.opt['gnl_omega_norm'] == 'tanh':
      self.Omega = torch.tanh(self.Omega)
    elif self.opt['gnl_omega_norm'] == 'rowSum':
      D = self.Omega.abs().sum(dim=1)
      self.Omega = torch.diag(torch.pow(D, -0.5)) @ self.Omega @ torch.diag(torch.pow(D, -0.5))
    else:
      pass
    return Omega

  def set_gnlOmega(self, T=None):
    if self.opt['gnl_omega'] == 'diag':
      if self.opt['gnl_omega_diag'] == 'free':
        # print(f"setting om_W {self.om_W.shape}")
        if self.time_dep_w:
          if T is None:
            T = 0
          Omega = torch.diag(self.om_W[T])
        else:
          Omega = torch.diag(self.om_W)
        # print(f"setting Omega{Omega.shape}")
        if self.opt['gnl_omega_activation'] == 'exponential':
          Omega = -torch.exp(Omega)
      elif self.opt['gnl_omega_diag'] == 'const':
        Omega = torch.diag(self.opt['gnl_omega_diag_val'] * torch.ones(self.in_features, device=self.device))
    elif self.opt['gnl_omega'] == 'zero':
      Omega = torch.zeros((self.in_features,self.in_features), device=self.device)
    elif self.opt['gnl_omega'] == 'Omega_eq_W':
      # broke
      Omega = -self.gnl_W
    elif self.opt['gnl_omega'] == 'Omega_W_eig':
      Omega = self.W_evec @ torch.diag(self.om_W) @ self.W_evec.T

    return Omega


  def set_gnlWS(self, T=None):
    if T is None:
      T = 0
    "note every W is made symetric before returning"
    if self.opt['gnl_W_style'] in ['prod']:
      return self.W_W @ self.W_W.t()
    elif self.opt['gnl_W_style'] in ['neg_prod']:
      return -self.W_W @ self.W_W.t()
    elif self.opt['gnl_W_style'] in ['sum']:
      return (self.W_W + self.W_W.t()) / 2
    elif self.opt['gnl_W_style'] in ['Z_diag']:
      return (self.W_W + self.W_W.t()) / 2
    elif self.opt['gnl_W_style'] == 'diag':
      if self.time_dep_w:
        if T is None:
          T = 0
        return torch.diag(self.gnl_W_D[T])
      elif self.time_dep_struct_w:
        if T is None:
          T = 0
        W = self.gnl_W_D[T]
        alpha = torch.diag(torch.exp(self.brt[T] * T + self.brt[T]))
        beta = torch.diag(torch.exp(-self.brt[T] * T - self.crt[T]) + self.drt[T])
        Wplus = torch.diag(F.relu(W))
        Wneg = torch.diag(-1. * F.relu(-W))
        return alpha @ Wplus - beta @ Wneg
      else:
        return torch.diag(self.gnl_W_D)
    elif self.opt['gnl_W_style'] == 'diag_dom':
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
    elif self.opt['gnl_W_style'] == 'k_block':
      W_temp = torch.cat([self.gnl_W_blocks, torch.zeros((self.opt['k_blocks'] * self.opt['block_size'], self.in_features - self.opt['block_size']), device=self.device)], dim=1)
      W_roll = torch.cat([torch.roll(W_temp[i:i+self.opt['block_size']], shifts=i*self.opt['block_size'], dims=1) for i in range(self.opt['k_blocks'])])
      W_zero_fill = torch.zeros(max(self.in_features - self.opt['block_size'] * self.opt['k_blocks'], 0), self.in_features, device=self.device)
      W = torch.cat((W_roll, W_zero_fill), dim=0)
      W[self.opt['k_blocks'] * self.opt['block_size']:,self.opt['k_blocks'] * self.opt['block_size']:] = torch.diag(self.gnl_W_D)
      Ws = (W+W.T) / 2
      return Ws
    elif self.opt['gnl_W_style'] == 'k_diag':
      W_temp = torch.cat([self.gnl_W_diags, torch.zeros((self.in_features, self.in_features - self.opt['k_diags']), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=int(i-(self.opt['k_diags']-1)/2), dims=-1) for i in range(self.in_features)])
      Ws = (W+W.T) / 2
      return Ws
    elif self.opt['gnl_W_style'] == 'k_diag_pc':
      W_temp = torch.cat([self.gnl_W_diags, torch.zeros((self.in_features, self.in_features - self.gnl_W_diags.shape[1]), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=int(i - (self.opt['k_diags'] - 1) / 2), dims=-1) for i in range(self.in_features)])
      Ws = (W + W.T) / 2
      return Ws
    elif self.opt['gnl_W_style'] in ['GS', 'GS_Z_diag']:
      self.V_hat = gram_schmidt(self.gnl_W_U)
      # W_D = torch.tanh(self.gnl_W_D) #      # W_D = torch.clamp(self.gnl_W_D, min=-1, max=1)
      # W_hat = V_hat @ torch.diag(W_D) @ V_hat.t()
      W_hat = self.V_hat @ torch.diag(self.gnl_W_D) @ self.V_hat.t()
      return W_hat
    elif self.opt['gnl_W_style'] in ['cgnn', 'cgnn_Z_diag']: # https://github.com/JRowbottomGit/ContinuousGNN/blob/85d47b0748a19e06e305c21e99e1dd03d36ad314/src/trainer.py
      beta = self.opt['W_beta']
      with torch.no_grad(): #todo need to check and undertand how/why this learns again
        W_U = self.gnl_W_U.clone()
        W_U = self.gnl_W_U.copy_((1 + beta) * W_U - beta * W_U @ W_U.t() @ W_U)
      # W_D = torch.clamp(self.gnl_W_D, min=-1, max=1) #self.gnl_W_D
      W_D = torch.tanh(self.gnl_W_D) #self.gnl_W_D
      W_hat = W_U @ torch.diag(W_D) @ W_U.t()
      return W_hat
    elif self.opt['gnl_W_style'] == 'identity':
      return torch.eye(self.in_features, device=self.device)
    # elif self.opt['W_type'] == 'residual_prod':
    #   return torch.eye(self.W.shape[0], device=x.device) + self.W @ self.W.t()  # output a [d,d] tensor
    elif self.opt['gnl_W_style'] == 'feature':
      pass
    elif self.opt['gnl_W_style'] == 'positional':
      pass

    elif self.opt['gnl_W_style'] == 'att_rep_lap_block':
      half_in_features = int(self.in_features / 2)
      L_temp = torch.cat([self.L_W, torch.zeros((half_in_features, 1), device=self.device)], dim=1)
      L = torch.stack([torch.roll(L_temp[i], shifts=i+1, dims=-1) for i in range(half_in_features)])
      L = (L+L.T) / 2
      # L_sum = (torch.exp(self.L_t_a) + 1) * torch.abs(L).sum(dim=1) + torch.exp(self.L_r_a)
      L_sum = (self.L_t_a**2 + 1) * torch.abs(L).sum(dim=1) + self.L_r_a**2
      L_block = L + torch.diag(L_sum)
      Ws = torch.zeros((self.in_features, self.in_features), device=self.device)
      Ws[0:half_in_features, 0:half_in_features] = L_block

      R_temp = torch.cat([self.R_W, torch.zeros((half_in_features, 1), device=self.device)], dim=1)
      R = torch.stack([torch.roll(R_temp[i], shifts=i+1, dims=-1) for i in range(half_in_features)])
      R = (R+R.T) / 2
      # R_sum = (torch.exp(self.R_t_a) + 1) * torch.abs(R).sum(dim=1) + torch.exp(self.R_r_a)
      R_sum = (self.R_t_a**2 + 1) * torch.abs(R).sum(dim=1) + self.R_r_a**2
      R_block = R + torch.diag(R_sum)
      R_Ws = torch.zeros((self.in_features, self.in_features), device=self.device)
      R_Ws[half_in_features:, half_in_features:] = R_block

      return Ws, R_Ws

  def get_energy_gradient(self, x, tau, tau_transpose, attentions, edge_index, n):
    row_sum = scatter_add(attentions, edge_index[0], dim=0, dim_size=n)
    deg_inv_sqrt = torch.pow(row_sum, -0.5)
    src_x, dst_x = self.get_src_dst(x)
    src_deg_inv_sqrt, dst_deg_inv_sqrt = self.get_src_dst(deg_inv_sqrt)
    src_term = (tau * src_x * src_deg_inv_sqrt.unsqueeze(dim=-1))
    dst_term = (tau_transpose * dst_x * dst_deg_inv_sqrt.unsqueeze(dim=-1))
    energy_gradient = (src_term - dst_term) @ self.W
    return energy_gradient

  def get_measure(self):
    # measure
    if self.opt['gnl_measure'] == 'deg_poly':
      measure = self.m_alpha * self.degree ** self.m_beta + self.m_gamma
      src_meas, dst_meas = self.get_src_dst(measure)
      measures_src_dst = 1 / (src_meas * dst_meas)
    elif self.opt['gnl_measure'] == 'nodewise':
      measure = self.measure
      src_meas, dst_meas = self.get_src_dst(measure)
      measures_src_dst = 1 / (src_meas * dst_meas)
    elif self.opt['gnl_measure'] == 'deg_poly_exp':
      measure = torch.exp(self.m_alpha * self.degree ** self.m_beta + self.m_gamma)
      src_meas, dst_meas = self.get_src_dst(measure)
      measures_src_dst = 1 / (src_meas * dst_meas)
    elif self.opt['gnl_measure'] == 'nodewise_exp':
      measure = torch.exp(self.measure)
      src_meas, dst_meas = self.get_src_dst(measure)
      measures_src_dst = 1 / (src_meas * dst_meas)
    elif self.opt['gnl_measure'] == 'ones':
      measure = torch.ones(self.n_nodes, device=self.device) #torch.tensor([1.]) #torch.ones(x.shape[0], device=self.device)
      src_meas = 1
      dst_meas = 1
      measures_src_dst = 1
    return measure, src_meas, dst_meas, measures_src_dst

  def do_diffusion(self, t):
    if self.opt['lie_trotter'] == 'gen_2':
      if self.opt['lt_block_type'] == 'diffusion' or self.opt['lt_block_type'] == 'label':
        return True
      else:
        return False
    else:
      if self.opt['drift']:
        return True
      if self.opt['lie_trotter'] in [None, 'gen_0']:
        return True
      if self.opt['lie_trotter'] == 'gen_1':
        for rng in self.opt['diffusion_ranges']:
          if t >= rng[0] and t < rng[1]:
            return True
    return False

  def do_drift(self, t):
    if self.opt['lie_trotter'] == 'gen_2':
      if self.opt['lt_block_type'] == 'drift':
        return True
      else:
        return False
    else:
      if self.opt['drift']:
        return True
      if self.opt['lie_trotter'] == 'gen_0':
        return True
      if self.opt['lie_trotter'] == 'gen_1':
        for rng in self.opt['drift_ranges']:
          if t >= rng[0] and t < rng[1]:
            return True
    return False

  def diffusion_step(self):
    pass

  def drift_step(self, x, f):
    # old style drift term
    if self.opt['drift_space'] == 'label':
      logits, pred = self.predict(x)
      sm_logits = torch.softmax(logits, dim=1)
      eye = torch.eye(self.C, device=self.device)
      dist_labels = sm_logits.unsqueeze(-1) - eye.unsqueeze(0)  # [num_nodes, c, 1] - [1, c, c]
      eta_hat = torch.sum(torch.abs(dist_labels), dim=1)  # sum abs distances for each node over features
      P = self.GNN_m2.weight
      index = list(range(self.C))
      for l in range(self.C):
        idx = index[:l] + index[l + 1:]
        q_l = eta_hat[:, l] * sm_logits[:, l]
        eta_l = torch.prod(eta_hat[:, idx] ** 2, dim=1) * q_l
        f -= (-torch.outer(eta_l, P[l]) + torch.outer(eta_l, torch.ones(sm_logits.shape[1], device=self.device)) * logits @ P) / torch.exp(self.drift_eps)

    # new style drift term
    elif self.opt['drift_space'] == 'feature':
      #todo system is stiff. product and sum of squared distances in R^d
      z_stack = torch.stack([z for z in self.attractors.values()], dim=1)
      dist_centers = x.unsqueeze(-1) - z_stack.unsqueeze(0)  # [num_nodes, d, 1] - [1, d, c]
      eta_hat = torch.sum(dist_centers**2, dim=1)  # sum abs distances for each node over features
      index = list(range(self.C))
      for l in range(self.C):
        idx = index[:l] + index[l + 1:]
        eta_l = torch.prod(eta_hat[:, idx], dim=1)
        z = self.attractors[l]
        f -= 0.5 * (torch.outer(eta_l, torch.ones(self.in_features, device=self.device)) *
                    (x - torch.outer(torch.ones(self.n_nodes, device=self.device), z))) / (torch.exp(self.drift_eps))
        #todo might need some feature regularisation or batch norm
    return f
  #todo augment potential to flatten extrema
  #todo argmin of distances to discrete threshold

  def predict(self, z):
    z = self.GNN_postXN(z)
    logits = self.GNN_m2(z)
    pred = logits.max(1)[1]
    return logits, pred

  def threshold(self, z, pred, step_size):
    #todo consider the interaction between decoder dropout, activation, augmentation and the inverse formula below
    # threshold in label space, pseudo inverse back to feature space
    Ek = F.one_hot(pred, num_classes=self.C)
    # pseudo inverse
    P = self.GNN_m2.weight
    b = self.GNN_m2.bias
    P_dagg = torch.linalg.pinv(P).T  # sometimes get RuntimeError: svd_cpu: the updating process of SBDSDC did not converge (error: 4)
    new_z = (Ek - b.unsqueeze(0)) @ P_dagg + z @ (torch.eye(P.shape[-1], device=self.device) - P_dagg.T @ P).T
    return (new_z - z) / step_size #returning value that will generate the change needed to get new_z (for explicit Euler)

  def calc_dot_prod_attention(self, src_x, dst_x):
    # scaled-dot method
    if self.opt['gnl_style'] == 'scaled_dot':
      fOmf = torch.einsum("ij,jk,ik->i", src_x, self.Omega, dst_x)

      if self.opt['gnl_activation'] == 'sigmoid':
        attention = torch.sigmoid(fOmf)
      elif self.opt['gnl_activation'] == "squareplus":
        attention = (fOmf + torch.sqrt(fOmf ** 2 + 4)) / 2
      elif self.opt['gnl_activation'] == "sigmoid_deriv":
        attention = sigmoid_deriv(fOmf)
      elif self.opt['gnl_activation'] == "tanh_deriv":
        attention = tanh_deriv(fOmf)
      elif self.opt['gnl_activation'] == "squareplus_deriv":
        attention = squareplus_deriv(fOmf)
      elif self.opt['gnl_activation'] == "exponential":
        attention = torch.exp(fOmf)
      elif self.opt['gnl_activation'] == 'identity':
        attention = fOmf
      else:
        attention = fOmf

    elif self.opt['gnl_style'] in ['general_graph', 'att_rep_laps']:#== 'general_graph':
      # get degrees
      src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)

      # calc bilinear form
      if not self.opt['gnl_activation'] == 'identity':
        fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                            dst_x * dst_deginvsqrt.unsqueeze(dim=1))
        # in the overleaf this is actually fWf just keeping for code homogeniety
        if self.opt['gnl_activation'] == 'sigmoid':
          attention = torch.sigmoid(fOmf)
        elif self.opt['gnl_activation'] == "squareplus":
          attention = (fOmf + torch.sqrt(fOmf ** 2 + 4)) / 2
        elif self.opt['gnl_activation'] == "sigmoid_deriv":
          attention = sigmoid_deriv(fOmf)
        elif self.opt['gnl_activation'] == "tanh_deriv":
          attention = tanh_deriv(fOmf)
        elif self.opt['gnl_activation'] == "squareplus_deriv":
          attention = squareplus_deriv(fOmf)
        elif self.opt['gnl_activation'] == "exponential":
          attention = torch.exp(fOmf)
        else:
          attention = fOmf
      elif self.opt['gnl_activation'] == 'identity':
        if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt[
          'wandb_epoch_list'] and self.get_evol_stats:  # not self.training:
          fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                              dst_x * dst_deginvsqrt.unsqueeze(dim=1))  # calc'd just for stats
        else:
          fOmf = torch.ones(src_deginvsqrt.shape, device=self.device)
        attention = torch.ones(src_deginvsqrt.shape, device=self.device)
    return fOmf, attention

  def set_M0(self):
    attention, _ = self.multihead_att_layer(self.x0, self.edge_index)
    # attention = sym_row_col(self.edge_index, attention, self.n_nodes) #already normalised in greed attention block
    self.mean_attention_0 = attention.mean(dim=1)

  def set_L0(self):
    # torch.sparse_coo_tensor(edges, self.L_0, (self.n_nodes, self.n_nodes)).to_dense().detach().numpy()
    A = torch.ones(self.edge_index.shape[1], device=self.device)
    A = self.symmetrically_normalise(A, self.edge_index)
    diags = torch.ones(self.n_nodes, device=self.device)
    L = torch.cat([-A, diags], dim=-1)
    self.L_0 = L

  def set_R0(self):
    A = torch.ones(self.edge_index.shape[1], device=self.device)
    A = self.symmetrically_normalise(A, self.edge_index)
    diags = torch.ones(self.n_nodes, device=self.device)
    R = torch.cat([-A, -diags], dim=-1)
    self.R_0 = R


  def forward(self, t, x):  # t is needed when called by the integrator
    self.paths.append(x) #append initial condition of the block

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    if self.time_dep_w:
      T = int(t / self.opt['step_size'])
      self.gnl_W = self.set_gnlWS(T)
      self.Omega = self.set_gnlOmega(T)
    else:
      T = 0

    if self.opt['beltrami']:
      pass
    else:
      measure, src_meas, dst_meas, measures_src_dst = self.get_measure()

      if self.do_diffusion(t):
        src_x, dst_x = self.get_src_dst(x)
        #scaled-dot method
        if self.opt['gnl_style'] == 'scaled_dot':
          fOmf, attention = self.calc_dot_prod_attention(src_x, dst_x)
          MThM = measures_src_dst * attention
          f = torch_sparse.spmm(self.edge_index, -MThM, x.shape[0], x.shape[0], x @ self.Omega)
          f = f - self.delta * x  # break point np.isnan(f.sum().detach().numpy())

        #softmax_attention method
        elif self.opt['gnl_style'] == 'softmax_attention':
          attention_h, _ = self.multihead_att_layer(x, self.edge_index)
          attention = attention_h.mean(dim=1)
          if self.opt['symmetric_QK']:
            self.Omega = self.multihead_att_layer.QK.weight.T @ self.multihead_att_layer.QK.weight
          else:
            self.Omega = (self.multihead_att_layer.Q.weight.T @ self.multihead_att_layer.K.weight +
                          self.multihead_att_layer.K.weight.T @ self.multihead_att_layer.Q.weight) / 2
          xOm = x @ self.Omega
          f1 = torch_sparse.spmm(self.edge_index, -attention / src_meas, x.shape[0], x.shape[0], xOm)
          index_t, att_t = transpose(self.edge_index, attention, x.shape[0], x.shape[0])
          f2 = torch_sparse.spmm(index_t, -att_t / dst_meas, x.shape[0], x.shape[0], xOm)
          f = f1 + f2
          f = f - self.delta * x  # break point np.isnan(f.sum().detach().numpy())

        #general graph (GCN/GraphSage) method
        elif self.opt['gnl_style'] == 'general_graph':
          fOmf, attention = self.calc_dot_prod_attention(src_x, dst_x)
          src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)
          P = attention * src_deginvsqrt * dst_deginvsqrt
          xW = x @ self.gnl_W
          if not self.opt['gnl_measure'] == 'ones':
            f1 = torch_sparse.spmm(self.edge_index, P / src_meas, x.shape[0], x.shape[0], xW) / 2
            f2 = torch_sparse.spmm(self.edge_index, P / dst_meas, x.shape[0], x.shape[0], xW) / 2
            f = f1 + f2
            f = f - torch.diag(1 / measure) @ x @ self.Omega
          else:
            if self.opt['gnl_attention']: #todo attention only implemented for measure==ones
              P = P * self.mean_attention_0
            f = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], xW)
            if self.opt['two_hops']:
              xWtilde = x @ self.gnl_W_tilde
              AA_ei, AA_val = torch_sparse.spspmm(self.edge_index, P, self.edge_index, P, x.shape[0], x.shape[0], x.shape[0])
              f = f - torch_sparse.spmm(AA_ei, AA_val, x.shape[0], x.shape[0], xWtilde) / 2
            f = f - x @ self.Omega

        #as per old_main.tex eq (11)
        elif self.opt['gnl_style'] == 'att_rep_laps':
          edges = torch.cat([self.edge_index, self.self_loops], dim=1)
          LfW = 0
          RfW = 0
          if self.opt['diffusion']:
            Ws = self.Ws
            fWs = torch.matmul(x, Ws)
            LfW = torch_sparse.spmm(edges, -self.L_0, x.shape[0], x.shape[0], fWs)
          if self.opt['repulsion']:
            R_Ws = self.R_Ws
            fRWs = torch.matmul(x, R_Ws)
            RfW = torch_sparse.spmm(edges, self.R_0, x.shape[0], x.shape[0], fRWs)

          try:
            self.alpha = float(self.opt['alpha_style'])
          except:
            if self.opt['alpha_style'] == "sigmoid":
              self.alpha = torch.sigmoid(self.alpha_train)
            elif self.opt['alpha_style'] == "free":
              self.alpha = self.alpha_train
            elif self.opt['alpha_style'] == "forced":
              self.alpha = self.opt['fix_alpha']
            elif self.opt['alpha_style'] == "diag":
              self.alpha = torch.diag(self.alpha_diag)
          if self.opt['alpha_style'] == "diag":
            if self.opt['diffusion'] and self.opt['repulsion']:
              f = torch.einsum("ij,kj->ki", self.alpha, LfW) + torch.einsum("ij,kj->ki", 1 - self.alpha, RfW)
            elif self.opt['diffusion'] and not self.opt['repulsion']:
              f = torch.einsum("ij,kj->ki", self.alpha, LfW)
            elif not self.opt['diffusion'] and self.opt['repulsion']:
              f = torch.einsum("ij,kj->ki", 1 - self.alpha, RfW)
          else:
            f = self.alpha * LfW + (1 - self.alpha) * RfW
            #torch.sparse_coo_tensor(edges, self.L_0, (self.n_nodes, self.n_nodes)).to_dense().detach().numpy()

        if self.opt['lie_trotter'] == 'gen_2':
          if self.opt['lt_block_type'] != 'label':
            if self.opt['test_mu_0']:
              if self.opt['add_source']:
                f = f + self.beta_train * self.x0
            else:
              f = f - 0.5 * self.mu * (x - self.x0)
        else:
          if self.opt['test_mu_0']:
            if self.opt['add_source']:
              f = f + self.beta_train * self.x0
          else:
            f = f - 0.5 * self.mu * (x - self.x0)
            # f = f - 0.5 * self.beta_train * (x - self.x0) #replacing beta with mu

      if self.do_drift(t):
        if not self.do_diffusion(t):
          f = torch.zeros(x.shape, device=self.device)
        f = torch.exp(self.drift_eps) * f
        x_temp = x
        if self.opt['lie_trotter'] == 'gen_0':
          x_temp = x_temp + self.opt['step_size'] * f  # take an euler step in diffusion direction

        if self.opt['drift_grad']:
          f = self.drift_step(x_temp, f)
        else:
          with torch.no_grad(): # todo understand what this means to not take gradient here
            f = self.drift_step(x_temp, f)

      if self.opt['gnl_thresholding'] and t in self.opt['threshold_times']:
        x = x + self.opt['step_size'] * f  #take an euler step that would have been taken from diff and dift gradients
        logits, pred = self.predict(x)
        f = self.threshold(x, pred, self.opt['step_size']) #generates change needed to snap to required value

    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.get_evol_stats:
      with torch.no_grad():
          fOmf, logits, attention, L2dist, train_acc, val_acc, test_acc, homophil, conf_mat, train_cm, val_cm, test_cm, \
          eval_means_feat, eval_sds_feat, eval_means_label, eval_sds_label, \
          entropies = generate_stats(self, t, x, f)

          append_stats(self, attention, fOmf, logits, x, measure, L2dist, train_acc, val_acc, test_acc, homophil, conf_mat,
                       train_cm, val_cm, test_cm,
                       eval_means_feat, eval_sds_feat, eval_means_label, eval_sds_label, entropies)

          ### extra values for terminal step
          # (only final block and it's not lie-trotter gen2 not final block) <- this is delt with in the "pass_stats" funtion in the LT2 block
          if t == self.opt['time'] - self.opt['step_size']:# and not(self.opt['lie_trotter'] == 'gen_2' and self.block_num + 1 != len(self.opt['lt_gen2_args'])):
            z = x + self.opt['step_size'] * f #take an euler step

            fOmf, logits, attention, L2dist, train_acc, val_acc, test_acc, homophil, conf_mat, train_cm, val_cm, test_cm, \
            eval_means_feat, eval_sds_feat, eval_means_label, eval_sds_label, \
            entropies = generate_stats(self, t, z, f) #f here is technically wrong but it only goes into energy calc

            append_stats(self, attention, fOmf, logits, z, measure, L2dist, train_acc, val_acc, test_acc, homophil, conf_mat,
                         train_cm, val_cm, test_cm,
                         eval_means_feat, eval_sds_feat, eval_means_label, eval_sds_label, entropies)
            if not(self.opt['lie_trotter'] == 'gen_2' and self.block_num + 1 != len(self.opt['lt_gen2_args'])):
              stack_stats(self) #todo move this to gnn level to make sure at the end

          self.wandb_step += 1

      # if self.opt['greed_momentum'] and self.prev_grad:
      #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      #   self.prev_grad = f
    return f - (1-self.opt['dampen_gamma']) * x /self.opt['step_size']

def __repr__(self):
  return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'