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
from torch_geometric.nn.inits import glorot, zeros, ones
from torch.nn import Parameter
import wandb
from function_greed import ODEFuncGreed
from utils import MaxNFEException, sym_row_col, sym_row_col_att, sym_row_col_att_measure, gram_schmidt
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer
from function_transformer_attention_greed import SpGraphTransAttentionLayer_greed


class ODEFuncGreedLinHet(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedLinHet, self).__init__(in_features, out_features, opt, data, device, bias=False)

    self.energy = 0
    self.attentions = None #torch.zeros((data.edge_index.shape[1],1))
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

      if opt['W_type'] == 'identity': #<- fix W s.t. W_s == I
        self.W = torch.cat([torch.eye(opt['hidden_dim']-opt['pos_enc_hidden_dim'], device=device), torch.zeros(opt['hidden_dim']-opt['pos_enc_hidden_dim'], max(opt['dim_p_w'] - opt['hidden_dim']-opt['pos_enc_hidden_dim'], 0), device=device)], dim=1)
      elif opt['W_type'] == 'diag':
        self.W = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim']))
      elif opt['W_type'] == 'residual':
        self.W = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim'], opt['dim_p_w']))
      elif opt['W_type'] == 'full':
        self.W = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim'], opt['dim_p_w']))
      elif opt['W_type'] == 'residual_GS':
        self.W_U = Parameter(torch.Tensor(opt['hidden_dim']-opt['pos_enc_hidden_dim'], opt['dim_W_k']))
        self.W_L = -torch.ones(opt['dim_W_k'], device=device)

    else:
      self.x_0 = None
      self.L_0 = None #static Laplacian form
      self.R_0 = None
      if not self.opt['test_tau_symmetric']:
        self.Q = Parameter(torch.Tensor(opt['hidden_dim'], 1))
      self.K = Parameter(torch.Tensor(opt['hidden_dim'], 1))

      if opt['W_type'] == 'identity': #<- fix W s.t. W_s == I
        self.W = torch.cat([torch.eye(in_features, device=device),
                            torch.zeros(in_features, max(opt['dim_p_w'] - in_features, 0), device=device)], dim=1)
      elif opt['W_type'] == 'diag':
        self.W = Parameter(torch.Tensor(in_features))
      elif opt['W_type'] == 'residual':
        self.W = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
      elif opt['W_type'] == 'full':
        self.W = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
      elif opt['W_type'] == 'full_idty':
        self.W = Parameter(torch.cat([torch.eye(in_features, device=device),
                            torch.zeros(in_features, max(opt['dim_p_w'] - in_features, 0), device=device)], dim=1))
      elif opt['W_type'] == 'residual_GS':
        self.W_U = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
        # self.W_L = -torch.ones(opt['dim_p_w'], device=device)
        self.W_L = Parameter(torch.Tensor(opt['dim_p_w']))

    self.measure = Parameter(torch.Tensor(self.n_nodes))
    self.C = (data.y.max()+1).item()
    self.attractors = {i: Parameter(torch.Tensor(opt['hidden_dim'])) for i in range(self.C)}

    self.reset_linH_parameters()

    # self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt, #check out_features is attention_dim
    #                                                       device, edge_weights=self.edge_weight).to(device)
    self.multihead_att_layer = SpGraphTransAttentionLayer_greed(in_features, out_features, opt, #check out_features is attention_dim
                                                          device, edge_weights=self.edge_weight).to(device)
    self.multihead_att_layer_R0 = SpGraphTransAttentionLayer_greed(in_features, out_features, opt, #check out_features is attention_dim
                                                          device, edge_weights=self.edge_weight).to(device)

  def reset_linH_parameters(self):
    if self.opt['beltrami']:
      if not self.opt['test_tau_symmetric']:
        glorot(self.Qx)
        glorot(self.Qp)
      glorot(self.Kx)
      glorot(self.Kp)
      # if not self.opt['test_no_chanel_mix']:  ##not having this might have been making Ws not identity for MANY cases
      #   glorot(self.W) #for non beltrami W init and reset in function_greed
    else:
      if not self.opt['test_tau_symmetric']:
        glorot(self.Q)
      glorot(self.K)
    zeros(self.bias)
    ones(self.measure)

    if self.opt['W_type'] == 'identity': #<- fix W s.t. W_s == I
      pass
    elif self.opt['W_type'] == 'diag':
      ones(self.W)
    elif self.opt['W_type'] == 'residual':
      zeros(self.W)
    elif self.opt['W_type'] == 'full':
      glorot(self.W)
    elif self.opt['W_type'] == 'full_idty':
      pass #todo figure out if this is really neccessary as bit fiddly to init as identity
    elif self.opt['W_type'] == 'residual_GS':
      glorot(self.W_U)
      zeros(self.W_L)

    if self.opt['drift']:
      for c in self.attractors.values():
        zeros(c)

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

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    values = torch.cat([-A, degree], dim=-1)

    if self.opt['laplacian_norm'] == "lap_symmDegnorm":
      L = self.symmetrically_normalise(values, edges)
    elif self.opt['laplacian_norm'] == "lap_symmRowSumnorm":
      L = sym_row_col(edges, values, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_symmAtt_RowSumnorm":
      L = sym_row_col_att(self.edge_index, A, edges, values, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_symmAttM_RowSumnorm":
      L = sym_row_col_att_measure(self.edge_index, A, edges, values, self.measure, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_noNorm":
      L = values

    return L

  def set_R0(self):
    attention, _ = self.multihead_att_layer_R0(self.x_0, self.edge_index)
    self.mean_attention_R0 = attention.mean(dim=1)

    if self.opt['test_omit_metric']:
      gamma = torch.ones(self.mean_attention_R0.shape, device=self.mean_attention_R0.device) #setting metric equal to adjacency
    else:
      gamma = self.mean_attention_R0

    if self.opt['beltrami']:
      self.Rf_0 = self.get_R0_linear(gamma, self.tau_f_0, self.tau_f_transpose_0)
      self.Rp_0 = self.get_R0_linear(gamma, self.tau_p_0, self.tau_p_transpose_0)
    else:
      self.R_0 = self.get_R0_linear(gamma, self.tau_0, self.tau_transpose_0)


  def get_R0_linear(self, gamma, tau, tau_transpose):
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
    R = self.get_R0_form_normed(T1, T0)
    return R


  # def get_laplacian_form_normed(self, T1, T0)
  def get_R0_form_normed(self, B, D):
    """
    Takes two sparse matrices A and D and performs sym_norm(D' - A) where D' is the degree matrix from row summing D
    @param A: Matrix that plays the role of the adjacency
    @param D: Matrix that is row summed to play the role of the degree matrix
    @return: A Laplacian form
    """
    if self.opt['R_T0term_normalisation'] == "T0_rowSum":
      degree = scatter_add(D, self.edge_index[0, :], dim=0, dim_size=self.n_nodes)
    elif self.opt['R_T0term_normalisation'] == "T0_identity":
      degree = torch.ones(self.n_nodes, device=D.device) #set this to ones to replicate good result from GRAND incremental

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    values = torch.cat([-B, -degree], dim=-1)

    if self.opt['R_laplacian_norm'] == "lap_symmDegnorm":
      R = self.symmetrically_normalise(values, edges)
    elif self.opt['R_laplacian_norm'] == "lap_symmRowSumnorm":
      R = sym_row_col(edges, values, self.n_nodes)
    elif self.opt['R_laplacian_norm'] == "lap_symmAtt_RowSumnorm":
      R = sym_row_col_att(self.edge_index, B, edges, values, self.n_nodes)
    elif self.opt['R_laplacian_norm'] == "lap_symmAttM_RowSumnorm":
      R = sym_row_col_att_measure(self.edge_index, B, edges, values, self.measure, self.n_nodes)
    elif self.opt['R_laplacian_norm'] == "lap_noNorm":
      R = values

    return R

  def set_GS(self):
    V_hat = gram_schmidt(self.W_U)
    W_hat = V_hat @ torch.diag(torch.exp(self.W_L) - 1.5) @ V_hat.t()
    self.Ws = torch.eye(self.W.shape[0], device=self.device) + W_hat

  def get_energy_gradient(self, x, tau, tau_transpose, attentions, edge_index, n):
    row_sum = scatter_add(attentions, edge_index[0], dim=0, dim_size=n)
    deg_inv_sqrt = torch.pow(row_sum, -0.5)
    src_x, dst_x = self.get_src_dst(x)
    src_deg_inv_sqrt, dst_deg_inv_sqrt = self.get_src_dst(deg_inv_sqrt)
    src_term = (tau * src_x * src_deg_inv_sqrt.unsqueeze(dim=-1))
    dst_term = (tau_transpose * dst_x * dst_deg_inv_sqrt.unsqueeze(dim=-1))
    energy_gradient = (src_term - dst_term) @ self.W
    return energy_gradient

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    if self.opt['W_type'] in ['identity', 'full', 'full_idty']:
      Ws = self.W @ self.W.t()  # output a [d,d] tensor
    elif self.opt['W_type'] == 'residual':
      Ws = torch.eye(self.W.shape[0], device=x.device) + self.W @ self.W.t()  # output a [d,d] tensor
    elif self.opt['W_type'] == 'diag':
      Ws = torch.diag(self.W)
    elif self.opt['W_type'] == 'residual_GS':
      #todo only need to do GS once at start of each epoch, ie at the forward pass of the GNN
      # V_hat = gram_schmidt(self.W_U)
      # W_hat = V_hat @ torch.diag(torch.exp(self.W_L) - 1.5) @ V_hat.t()
      # Ws = torch.eye(self.W.shape[0], device=x.device) + W_hat
      Ws = self.Ws

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)

    if self.opt['beltrami']:
      # x is [(features, pos_encs) * aug_factor, lables] but it's safe to assume aug_factor == 1
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      xf = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)
      ff = torch_sparse.spmm(edges, -self.Lf_0, xf.shape[0], xf.shape[0], xf)
      ff = torch.matmul(ff, Ws)
      ff = ff - self.mu * (xf - self.xf_0)
      fp = torch_sparse.spmm(edges, -self.Lp_0, p.shape[0], p.shape[0], p)
      f = torch.cat([ff, fp], dim=1) #assuming don't have any augmentation or labels

    else:
      Lf = torch_sparse.spmm(edges, -self.L_0, x.shape[0], x.shape[0], x)
      LfW = torch.matmul(Lf, Ws)

      if not self.opt['no_alpha_sigmoid']:
        alpha = torch.sigmoid(self.alpha_train)
      else:
        alpha = self.alpha_train

      if self.opt['repulsion']:
        Rf = torch_sparse.spmm(edges, self.R_0, x.shape[0], x.shape[0], x) #LpRf
        RfW = torch.matmul(Rf, Ws) #todo need a second Ws
        f = alpha * LfW + (1-alpha) * RfW
        # f = 0.9 * LfW + (1-0.9) * RfW
      else:
        f = alpha * LfW

    if self.opt['test_mu_0']:
      if self.opt['add_source']:
        f = f + self.beta_train * self.x0
    else:
      f = f - 0.5 * self.mu * (x - self.x0)
      # f = f - 0.5 * self.beta_train * (x - self.x0) #replacing beta with mu

    if self.opt['drift']:
      drift = -self.C * f
      for c in self.attractors.values():
        drift += c
      f = f + drift

    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.training:
      energy = torch.sum(
        self.get_energy_gradient(x, self.tau_0, self.tau_transpose_0, self.mean_attention_0, self.edge_index,
                                 self.n_nodes) ** 2)
      if self.opt['drift']:
        energy = energy
      if self.opt['repulsion']:
        energy = energy
      if self.opt['test_mu_0'] and self.opt['add_source']:
        energy = energy - self.beta_train * torch.sum(x * self.x0)
      elif not self.opt['test_mu_0']:
        energy = energy + self.mu * torch.sum((x - self.x0) ** 2)
      else:
        # with torch.no_grad(): #these things only go into calculating Energy not forward
        #   metric_0 = self.get_metric(self.x_0, self.tau_0, self.tau_transpose_0)
        #   _, eta = self.get_gamma(metric_0, self.opt['gamma_epsilon'])
        #   tau, tau_transpose = self.tau_0, self.tau_transpose_0 #assigning for energy calcs
        # energy = self.get_energy(x, eta)
        energy = 0
        self.energy = energy

      wandb.log({f"gf_e{self.epoch}_energy_change": energy - self.energy, f"gf_e{self.epoch}_energy": energy,
                 f"gf_e{self.epoch}_f": (f**2).sum(),
                 "grad_flow_step": self.wandb_step})
      if self.attentions is None:
        self.attentions = self.mean_attention_0
      else:
        self.attentions = torch.cat([self.attentions, self.mean_attention_0], dim=-1)
      # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
      # I think I need to build the attentions tensor a save at the end
      # f"gf_e{self.epoch}_attentions": wandb.plot.line_series(xs=self.wandb_step, ys=self.mean_attention_0),

      self.wandb_step += 1

    # if self.opt['greed_momentum'] and self.prev_grad:
    #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
    #   self.prev_grad = f

    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
