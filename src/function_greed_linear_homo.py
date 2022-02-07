"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
import numpy as np
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import degree, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
import wandb
from function_greed import ODEFuncGreed
from utils import MaxNFEException, sym_row_col
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer
from function_transformer_attention_greed import SpGraphTransAttentionLayer_greed


class ODEFuncGreedLinH(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedLinH, self).__init__(in_features, out_features, opt, data, device, bias=False)

    self.energy = 0
    self.epoch = 0
    self.wandb_step = 0
    self.prev_grad = None

    if self.opt['beltrami']:
      self.xf_0 = None
      self.xp_0 = None
      self.Lf_0 = None #static Laplacian form
      self.Lp_0 = None #static Laplacian form
      if not self.opt['test_tau_symmetric']:
        self.Qx = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim'], 1))
        self.Qp = Parameter(torch.Tensor(opt['pos_enc_hidden_dim'], 1))
      self.Kx = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim'], 1))
      self.Kp = Parameter(torch.Tensor(opt['pos_enc_hidden_dim'], 1))

      if opt['test_no_chanel_mix']: #<- fix W s.t. W_s == I
        self.W = torch.cat([torch.eye(opt['hidden_dim']-opt['pos_enc_hidden_dim'], device=device), torch.zeros(opt['hidden_dim']-opt['pos_enc_hidden_dim'], max(opt['dim_p_w'] - opt['hidden_dim']-opt['pos_enc_hidden_dim'], 0), device=device)], dim=1)
      else:
        self.W = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim'], opt['dim_p_w']))
    else:
      self.x_0 = None
      self.L_0 = None #static Laplacian form
      if not self.opt['test_tau_symmetric']:
        self.Q = Parameter(torch.Tensor(opt['hidden_dim'], 1))
      self.K = Parameter(torch.Tensor(opt['hidden_dim'], 1))

    self.reset_linH_parameters()

    # self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt, #check out_features is attention_dim
    #                                                       device, edge_weights=self.edge_weight).to(device)
    self.multihead_att_layer = SpGraphTransAttentionLayer_greed(in_features, out_features, opt, #check out_features is attention_dim
                                                          device, edge_weights=self.edge_weight).to(device)

  def reset_linH_parameters(self):
    if self.opt['beltrami']:
      if not self.opt['test_tau_symmetric']:
        glorot(self.Qx)
        glorot(self.Qp)
      glorot(self.Kx)
      glorot(self.Kp)
      if not self.opt['test_no_chanel_mix']:  ##not having this might have been making Ws not identity for MANY cases
        glorot(self.W) #for non beltrami W init and reset in function_greed
    else:
      if not self.opt['test_tau_symmetric']:
        glorot(self.Q)
      glorot(self.K)
    zeros(self.bias)



  def set_x_0(self, x_0):
    self.x_0 = x_0.clone().detach()
    if self.opt['beltrami']:
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p_0 = x_0[:, self.opt['feat_hidden_dim']: label_index]
      xf_0 = torch.cat((x_0[:, :self.opt['feat_hidden_dim']], x_0[:, label_index:]), dim=1)
      self.xf_0 = xf_0.clone().detach()
      self.p_0 = p_0.clone().detach()


  def get_tau(self, x, Q, K):
    src_x, dst_x = self.get_src_dst(x)
    if self.opt['test_tau_symmetric']:
      tau = torch.tanh((src_x + dst_x) @ K / self.opt['tau_reg'])
      tau_transpose = torch.tanh((dst_x + src_x) @ K / self.opt['tau_reg'])
    else:
      tau = torch.tanh((src_x @ K + dst_x @ Q) / self.opt['tau_reg'])
      tau_transpose = torch.tanh((dst_x @ K + src_x @ Q) / self.opt['tau_reg'])
    if self.opt['test_tau_ones']:
      tau = torch.ones(tau.shape, device=tau.device)
      tau_transpose = torch.ones(tau_transpose.shape, device=tau_transpose.device)

    return tau, tau_transpose


  def set_tau_0(self):
    if self.opt['beltrami']:
      if self.opt['test_tau_symmetric']:
        self.tau_f_0, self.tau_f_transpose_0 = self.get_tau(self.xf_0, self.Kx, self.Kx)
        self.tau_p_0, self.tau_p_transpose_0 = self.get_tau(self.p_0, self.Kp, self.Kp)
      else:
        self.tau_f_0, self.tau_f_transpose_0 = self.get_tau(self.xf_0, self.Qx, self.Kx)
        self.tau_p_0, self.tau_p_transpose_0 = self.get_tau(self.p_0, self.Qp, self.Kp)
    else:
      if self.opt['test_tau_symmetric']:
        self.tau_0, self.tau_transpose_0 = self.get_tau(self.x_0, self.K, self.K)
      else:
        self.tau_0, self.tau_transpose_0 = self.get_tau(self.x_0, self.Q, self.K)


  def set_L0(self):
    attention, _ = self.multihead_att_layer(self.x_0, self.edge_index)
    self.mean_attention_0 = attention.mean(dim=1)

    if self.opt['test_omit_metric']:
      gamma = torch.ones(self.mean_attention_0.shape, device=self.mean_attention_0.device) #setting metric equal to adjacency
    else:
      gamma = self.mean_attention_0

    if self.opt['beltrami']:
      self.Lf_0 = self.get_laplacian_linear(gamma, self.tau_f_0, self.tau_f_transpose_0)
      self.Lp_0 = self.get_laplacian_linear(gamma, self.tau_p_0, self.tau_p_transpose_0)
    else:
      self.L_0 = self.get_laplacian_linear(gamma, self.tau_0, self.tau_transpose_0)


  def get_laplacian_linear(self, gamma, tau, tau_transpose):
    """
    These dynamics follow directly from differentiating the energy
    The structure of T0,T1 etc. is a direct result of differentiating the hyperbolic tangent nonlinearity.
    Note: changing the nonlinearity will change the equations
    Returns: L - a generalisation of the Laplacian, but using attention
             R1, R2 are a result of tau being a function of the features and both are zero if tau is not dependent on
             features
    """
    tau = tau.flatten()
    tau_transpose = tau_transpose.flatten()
    tau2 = tau * tau
    T0 = gamma * tau2
    T1 = gamma * tau * tau_transpose
    L = self.get_laplacian_form_normed(T1, T0)
    return L

  # def get_laplacian_form_normed(self, T1, T0)
  def get_laplacian_form_normed(self, A, D):
    """
    Takes two sparse matrices A and D and performs sym_norm(D' - A) where D' is the degree matrix from row summing D
    @param A: Matrix that plays the role of the adjacency
    @param D: Matrix that is row summed to play the role of the degree matrix
    @return: A Laplacian form
    """
    if self.opt['T0term_normalisation'] == "T0_rowSum":
      degree = scatter_add(D, self.edge_index[0, :], dim=0, dim_size=self.n_nodes)
    elif self.opt['T0term_normalisation'] == "T0_identity":
      degree = torch.ones(self.n_nodes, device=D.device) #set this to ones to replicate good result from GRAND incremental

    # if self.opt['T1term_normalisation'] == "T1_symmDegnorm": #like A in A_hat = A - I
    #   A = self.symmetrically_normalise(A, self.self_loops)
    # elif self.opt['T1term_normalisation'] == "T1_symmRowSumnorm":
    #   A = sym_row_col(self.edge_index, A, self.n_nodes)
    if self.opt['T1term_normalisation'] == "T1_noNorm":
      pass

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    values = torch.cat([-A, degree], dim=-1)

    if self.opt['laplacian_norm'] == "lap_symmDegnorm":
      L = self.symmetrically_normalise(values, edges)
    elif self.opt['laplacian_norm'] == "lap_symmRowSumnorm":
      L = sym_row_col(edges, values, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_noNorm":
      L = values

    return L


  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    Ws = self.W @ self.W.t()  # output a [d,d] tensor
    edges = torch.cat([self.edge_index, self.self_loops], dim=1)

    if self.opt['beltrami']:
      # x is [(features, pos_encs) * aug_factor, lables] but it's safe to assume aug_factor == 1
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      xf = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)
      ff = torch_sparse.spmm(edges, -self.Lf_0, xf.shape[0], xf.shape[0], xf)
      ff = torch.matmul(ff, Ws)
      ff = ff - self.mu * (xf - self.xf_0)
      # ff = ff - self.mu * (ff - self.xf_0)  #this is WRONG but the above doesn't work
      fp = torch_sparse.spmm(edges, -self.Lp_0, p.shape[0], p.shape[0], p)
      f = torch.cat([ff, fp], dim=1) #assuming don't have any augmentation or labels

    else:
      Lf = torch_sparse.spmm(edges, -self.L_0, x.shape[0], x.shape[0], x)
      LfW = torch.matmul(Lf, Ws)

      if not self.opt['no_alpha_sigmoid']:
        alpha = torch.sigmoid(self.alpha_train)
      else:
        alpha = self.alpha_train

      f = alpha * LfW

      if self.opt['test_mu_0']:
        if self.opt['add_source']:
          f = f + self.beta_train * self.x0
      else:
        # f = f + self.beta_train * self.x0
        # f = f - 0.5 * self.mu * (x - self.x0)
        f = f - 0.5 * self.beta_train * (x - self.x0) #replacing beta with mu

    # with torch.no_grad(): #these things only go into calculating Energy not forward
    #   metric_0 = self.get_metric(self.x_0, self.tau_0, self.tau_transpose_0)
    #   _, eta = self.get_gamma(metric_0, self.opt['gamma_epsilon'])
    #   tau, tau_transpose = self.tau_0, self.tau_transpose_0 #assigning for energy calcs

    ##NOT USED AS NOT TRACKING CURRENTLY
    # if self.opt['test_omit_metric'] and self.opt['test_mu_0']: #energy to use when Gamma is -adjacency and not the pullback and mu == 0
    #   energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2)
    # elif self.opt['test_omit_metric']: #energy to use when Gamma is -adjacency and not the pullback and mu != 0
    #   energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2) + self.mu * torch.sum((x - self.x0) ** 2)
    # else:
    #   energy = self.get_energy(x, eta)
    # R1 = 0
    # R2 = 0
    # energy = 0
    # if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.training:
    #   wandb.log({f"gf_e{self.epoch}_energy_change": energy - self.energy, f"gf_e{self.epoch}_energy": energy,
    #              f"gf_e{self.epoch}_f": f ** 2, f"gf_e{self.epoch}_L": torch.sum(L ** 2),
    #              f"gf_e{self.epoch}_R1": torch.sum(R1 ** 2), f"gf_e{self.epoch}_R2": torch.sum(R2 ** 2), f"gf_e{self.epoch}_mu": self.mu,
    #              "grad_flow_step": self.wandb_step})
    #   self.wandb_step += 1
    # self.energy = energy

    # if self.opt['greed_momentum'] and self.prev_grad:
    #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
    #   self.prev_grad = f

    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
