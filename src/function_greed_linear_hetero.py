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
from utils import MaxNFEException, sym_row_col, sym_row_col_att, sym_row_col_att_measure, gram_schmidt, sym_row_col_att_relaxed
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
      self.L_0 = 0  # static Laplacian form
      self.R_0 = 0

      if not self.opt['test_tau_symmetric']:
        self.Q = Parameter(torch.Tensor(opt['hidden_dim'], 1))
      self.K = Parameter(torch.Tensor(opt['hidden_dim'], 1))

      if opt['diffusion']:
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
          # self.W_L = -torch.ones(opt['dim_p_w'], device=device) #force this when heterophillic
          self.W_L = Parameter(torch.Tensor(opt['dim_p_w']))
        elif self.opt['W_type'] == 'cgnn':
          self.Ws = Parameter(torch.eye(in_features))
        elif self.opt['W_type'] in ['lin_layer', 'res_lin_layer', 'lin_layer_hh', 'res_layer_hh'
                                    ,'lin_layer_mean', 'res_lin_layer_mean', 'lin_layer_hh_mean', 'res_layer_hh_mean']:
          self.Ws_lin = nn.Linear(in_features, in_features**2, bias=False)
        elif self.opt['W_type'] in ['lin_layer_hp', 'res_layer_hp', 'lin_layer_hp_mean', 'res_layer_hp_mean']:
          self.Ws_lin = nn.Linear(in_features, in_features * opt['dim_p_w'], bias=False)
          self.W_L = -torch.ones(opt['dim_p_w'], device=device) #force this when heterophillic
          # self.W_L = Parameter(torch.Tensor(opt['dim_p_w']))

        self.multihead_att_layer = SpGraphTransAttentionLayer_greed(in_features, out_features, opt,
                                                                    # check out_features is attention_dim
                                                                    device, edge_weights=self.edge_weight).to(device)

      if opt['repulsion']:
        if opt['R_W_type'] == 'identity': #<- fix W s.t. W_s == I
          self.R_W = torch.cat([torch.eye(in_features, device=device),
                              torch.zeros(in_features, max(opt['dim_p_w'] - in_features, 0), device=device)], dim=1)
        elif opt['R_W_type'] == 'diag':
          self.R_W = Parameter(torch.Tensor(in_features))
        elif opt['R_W_type'] == 'residual':
          self.R_W = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
        elif opt['R_W_type'] == 'full':
          self.R_W = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
        elif opt['R_W_type'] == 'full_idty':
          self.R_W = Parameter(torch.cat([torch.eye(in_features, device=device),
                              torch.zeros(in_features, max(opt['dim_p_w'] - in_features, 0), device=device)], dim=1))
        elif opt['R_W_type'] == 'residual_GS':
          self.R_W_U = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
          # self.W_L = -torch.ones(opt['dim_p_w'], device=device) #force this when heterophillic
          self.R_W_L = Parameter(torch.Tensor(opt['dim_p_w']))
        elif self.opt['R_W_type'] == 'cgnn':
          self.R_Ws = Parameter(torch.eye(in_features))
        elif self.opt['R_W_type'] in ['lin_layer', 'res_lin_layer', 'lin_layer_hh', 'res_layer_hh'
                                    ,'lin_layer_mean', 'res_lin_layer_mean', 'lin_layer_hh_mean', 'res_layer_hh_mean']:
          self.R_Ws_lin = nn.Linear(in_features, in_features**2, bias=False)
        elif self.opt['R_W_type'] in ['lin_layer_hp', 'res_layer_hp', 'lin_layer_hp_mean', 'res_layer_hp_mean']:
          self.R_Ws_lin = nn.Linear(in_features, in_features * opt['dim_p_w'], bias=False)
          self.R_W_L = -torch.ones(opt['dim_p_w'], device=device) #force this when heterophillic
          # self.W_L = Parameter(torch.Tensor(opt['dim_p_w']))

        self.multihead_att_layer_R0 = SpGraphTransAttentionLayer_greed(in_features, out_features, opt, #check out_features is attention_dim
                                                              device, edge_weights=self.edge_weight).to(device)

    self.tau_l = Parameter(torch.Tensor(1)) #for residual tau
    self.measure = Parameter(torch.Tensor(self.n_nodes)) # used as either dividitive or additive nodewise normalisation measure
    self.C = (data.y.max()+1).item() #num class for drift
    self.attractors = {i: Parameter(torch.Tensor(opt['hidden_dim'])) for i in range(self.C)}
    self.alpha = 0.0
    self.reset_linH_parameters()


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
    ones(self.tau_l)

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
    elif self.opt['W_type'] == 'cgnn':
      pass #self.Ws
    elif self.opt['W_type'] in ['lin_layer', 'res_lin_layer', 'lin_layer_hh', 'res_layer_hh'
                                  ,'lin_layer_mean', 'res_lin_layer_mean', 'lin_layer_hh_mean',
                                'res_layer_hh_mean', 'lin_layer_hp', 'res_layer_hp',
                                'lin_layer_hp_mean', 'res_layer_hp_mean']:
      nn.init.xavier_uniform_(self.Ws_lin.weight, gain=1.414)
      # nn.init.constant_(self.Ws.weight, 1e-1)#5)


    if self.opt['R_W_type'] == 'identity': #<- fix W s.t. W_s == I
      pass
    elif self.opt['R_W_type'] == 'diag':
      ones(self.R_W)
    elif self.opt['R_W_type'] == 'residual':
      zeros(self.R_W)
    elif self.opt['R_W_type'] == 'full':
      glorot(self.R_W)
    elif self.opt['R_W_type'] == 'full_idty':
      pass #todo figure out if this is really neccessary as bit fiddly to init as identity
    elif self.opt['R_W_type'] == 'residual_GS':
      glorot(self.R_W_U)
      zeros(self.R_W_L)
    elif self.opt['R_W_type'] == 'cgnn':
      pass #self.Ws
    elif self.opt['R_W_type'] in ['lin_layer', 'res_lin_layer', 'lin_layer_hh', 'res_layer_hh'
                                  ,'lin_layer_mean', 'res_lin_layer_mean', 'lin_layer_hh_mean',
                                'res_layer_hh_mean', 'lin_layer_hp', 'res_layer_hp',
                                'lin_layer_hp_mean', 'res_layer_hp_mean']:
      nn.init.xavier_uniform_(self.R_Ws_lin.weight, gain=1.414)
      # nn.init.constant_(self.Ws.weight, 1e-1)#5)

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
      tau = (src_x + dst_x) @ K / self.opt['tau_reg']
      tau_transpose = (dst_x + src_x) @ K / self.opt['tau_reg']
    else:
      tau = (src_x @ K + dst_x @ Q) / self.opt['tau_reg']
      tau_transpose = (dst_x @ K + src_x @ Q) / self.opt['tau_reg']

    if self.opt['tau_residual']:
      tau = tau + self.tau_l
      tau_transpose = tau_transpose + self.tau_l

    if not self.opt['test_tau_remove_tanh']:
      tau = torch.tanh(tau)
      tau_transpose = torch.tanh(tau_transpose)

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
      self.Lf_0 = self.get_laplacian_linear(gamma, self.tau_f_0, self.tau_f_transpose_0, form_type="diffusion")
      self.Lp_0 = self.get_laplacian_linear(gamma, self.tau_p_0, self.tau_p_transpose_0, form_type="diffusion")
    else:
      self.L_0 = self.get_laplacian_linear(gamma, self.tau_0, self.tau_transpose_0, form_type="diffusion")


  def set_R0(self):
    attention, _ = self.multihead_att_layer_R0(self.x_0, self.edge_index)
    self.mean_attention_R0 = attention.mean(dim=1)
    if self.opt['test_omit_metric']:
      gamma = torch.ones(self.mean_attention_R0.shape, device=self.mean_attention_R0.device) #setting metric equal to adjacency
    else:
      gamma = self.mean_attention_R0

    if self.opt['R_depon_A'] == 'decay':
      gamma = gamma * torch.exp(-self.mean_attention_0)
    elif self.opt['R_depon_A'] == 'inverse':
      gamma = 1 / (1 + self.mean_attention_0)

    if self.opt['beltrami']:
      self.Rf_0 = self.get_laplacian_linear(gamma, self.tau_f_0, self.tau_f_transpose_0, form_type="repulsion")
      self.Rp_0 = self.get_laplacian_linear(gamma, self.tau_p_0, self.tau_p_transpose_0, form_type="repulsion")
    else:
      self.R_0 = self.get_laplacian_linear(gamma, self.tau_0, self.tau_transpose_0, form_type="repulsion")


  def get_laplacian_linear(self, gamma, tau, tau_transpose, form_type):
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
    L = self.get_laplacian_form_normed(T1, T0, form_type)
    return L


  # def get_laplacian_form_normed(self, T1, T0)
  def get_laplacian_form_normed(self, A, D, form_type):
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
    if form_type == "diffusion":
      values = torch.cat([-A, degree], dim=-1)
    elif form_type == "repulsion":
      values = torch.cat([-A, -degree], dim=-1)

    if self.opt['laplacian_norm'] == "lap_symmDegnorm":
      L = self.symmetrically_normalise(values, edges)
    elif self.opt['laplacian_norm'] == "lap_symmRowSumnorm":
      L = sym_row_col(edges, values, self.n_nodes)

    elif self.opt['laplacian_norm'] == "lap_symmAtt_RowSumnorm":
      L = sym_row_col_att(self.edge_index, A, edges, values, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_symmAttM_RowSumnorm":
      L = sym_row_col_att_measure(self.edge_index, A, edges, values, self.measure, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_symmAtt_relaxed":
      L = sym_row_col_att_relaxed(self.edge_index, A, edges, values, self.measure, self.n_nodes)

    elif self.opt['laplacian_norm'] == "lap_symmDeg_RowSumnorm":
      L = sym_row_col_att(self.self_loops, degree, edges, values, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_symmDegM_RowSumnorm":
      L = sym_row_col_att_measure(self.self_loops, degree, edges, values, self.measure, self.n_nodes)
    elif self.opt['laplacian_norm'] == "lap_symmDeg_relaxed":
      L = sym_row_col_att_relaxed(self.self_loops, degree, edges, values, self.measure, self.n_nodes)

    elif self.opt['laplacian_norm'] == "lap_noNorm":
      L = values

    return L


  def set_WS(self, x):
    if self.opt['W_type'] in ['identity', 'full', 'full_idty']:
      return self.W @ self.W.t()  # output a [d,d] tensor
    elif self.opt['W_type'] == 'residual':
      return torch.eye(self.W.shape[0], device=x.device) + self.W @ self.W.t()  # output a [d,d] tensor
    elif self.opt['W_type'] == 'diag':
      return torch.diag(self.W)

    elif self.opt['W_type'] == 'residual_GS':
      V_hat = gram_schmidt(self.W_U)
      W_hat = V_hat @ torch.diag(torch.exp(self.W_L) - 1.5) @ V_hat.t()
      return torch.eye(self.W.shape[0], device=self.device) + W_hat

    elif self.opt['W_type'] == 'cgnn':
      beta = self.opt['W_beta']
      with torch.no_grad():
        Ws = self.Ws.clone()
        #todo check update should be on U
        # self.Ws.copy_((1 + beta) * Ws - beta * Ws @ Ws.t() @ Ws)

    elif self.opt['W_type'] == 'lin_layer':
      return self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]) #n,h,h -> Ws

    elif self.opt['W_type'] == 'lin_layer_hh':
      W = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]) #n,h,h -> W @ W.t
      ### XXXX
      # self.Ws = W @ W.t()

    elif self.opt['W_type'] == 'lin_layer_hp':
      W_U = self.Ws_lin(x).view(-1, x.shape[1], self.opt['dim_p_w']) #n,h,p -> U L U.t
      V_hat = gram_schmidt(W_U)
      ### XXXX
      # self.Ws = V_hat @ self.W_L @ V_hat.t()

    elif self.opt['W_type'] == 'res_lin_layer':
      W_hat = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]) #n,h,h -> Ws
      return torch.eye(self.W.shape[0], device=self.device) + W_hat
    elif self.opt['W_type'] == 'res_layer_hh':
      W = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]) #n,h,h -> W @ W.t
      ### XXXX
      W_hat = W @ W.t()
      # self.Ws = torch.eye(self.W.shape[0], device=self.device) + W_hat

    elif self.opt['W_type'] == 'res_layer_hp':
      W_U = self.Ws_lin(x).view(-1, x.shape[1], self.opt['dim_p_w']) #n,h,p -> U L U.t
      V_hat = gram_schmidt(W_U)
      ### XXXX
      W_hat = V_hat @ torch.diag(torch.exp(self.W_L) - 1.0) @ V_hat.t()
      # self.Ws = torch.eye(self.W.shape[0], device=self.device) + W_hat

    # mean
    elif self.opt['W_type'] == 'lin_layer_mean':
      return self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]).mean(dim=0) #h,h -> Ws
    elif self.opt['W_type'] == 'lin_layer_hh_mean':
      W = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]).mean(dim=0) #h,h -> W @ W.t
      return W @ W.t()
    elif self.opt['W_type'] == 'lin_layer_hp_mean':
      W_U = self.Ws_lin(x).view(-1, x.shape[1], self.opt['dim_p_w']).mean(dim=0) #h,p -> U L U.t
      V_hat = gram_schmidt(W_U)
      return V_hat @ torch.diag(self.W_L) @ V_hat.t()

    elif self.opt['W_type'] == 'res_lin_layer_mean':
      W_hat = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]).mean(dim=0) #h,h -> Ws
      return torch.eye(self.W.shape[0], device=self.device) + W_hat
    elif self.opt['W_type'] == 'res_layer_hh_mean':
      W = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]).mean(dim=0) #h,h -> W @ W.t
      W_hat = W @ W.t()
      return torch.eye(self.W.shape[0], device=self.device) + W_hat
    elif self.opt['W_type'] == 'res_layer_hp_mean':
      W_U = self.Ws_lin(x).view(-1, x.shape[1], self.opt['dim_p_w']).mean(dim=0) #h,p -> U L U.t
      V_hat = gram_schmidt(W_U)
      W_hat = V_hat @ torch.diag(torch.exp(self.W_L) - 1.0) @ V_hat.t()
      return torch.eye(self.W.shape[0], device=self.device) + W_hat

    # ['lin_layer', 'lin_layer_hh', 'lin_layer_hp', 'res_lin_layer', 'res_layer_hh', 'res_layer_hp']
    # ['lin_layer_mean', 'lin_layer_hh_mean', 'lin_layer_hp_mean',
    # 'res_lin_layer_mean', 'res_layer_hh_mean', 'res_layer_hp_mean']

    elif self.opt['W_type'] == 'QK_W':
      #exp ( < W_K f_i, W_Q f_j > )
      #W_K ^T * W_Q + W_Q ^T * W_K
      W_U = self.Ws_lin(x).view(-1, x.shape[1], x.shape[1]).mean(dim=0)

#todo
  # hyper parameter tune for heterophillic
  # average linear layer for W_U
  # QK_W dependnce for f_0 or vice versa
  # forced -1 lambda for hetrero phillic or free and report, check ULU.t s.t. AX=-X


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

    # Ws = self.Ws
    edges = torch.cat([self.edge_index, self.self_loops], dim=1)

    if self.opt['beltrami']:
      Ws = self.Ws
      # x is [(features, pos_encs) * aug_factor, lables] but here assume aug_factor == 1
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      xf = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)
      ff = torch_sparse.spmm(edges, -self.Lf_0, xf.shape[0], xf.shape[0], xf)
      ff = torch.matmul(ff, Ws)
      ff = ff - self.mu * (xf - self.xf_0)
      fp = torch_sparse.spmm(edges, -self.Lp_0, p.shape[0], p.shape[0], p)
      f = torch.cat([ff, fp], dim=1)

    else:
      LfW = 0
      RfW = 0
      if self.opt['diffusion']:
        Ws = self.Ws
        Lf = torch_sparse.spmm(edges, -self.L_0, x.shape[0], x.shape[0], x)
        if self.opt['W_type'] in ['lin_layer', 'lin_layer_hh', 'lin_layer_hp', 'res_lin_layer', 'res_layer_hh', 'res_layer_hp']:
          LfW = torch.einsum("ij,ikj->ik", Lf, Ws)
        else:
          LfW = torch.matmul(Lf, Ws)

      if self.opt['fix_alpha'] and (not self.opt['fix_alpha'] == "None" or not self.opt['fix_alpha'] == None):
        self.alpha = self.opt['fix_alpha']
      else:
        if not self.opt['no_alpha_sigmoid']:
          self.alpha = torch.sigmoid(self.alpha_train)
        else:
          self.alpha = self.alpha_train

      if self.opt['repulsion']:
        R_Ws = self.R_Ws
        Rf = torch_sparse.spmm(edges, self.R_0, x.shape[0], x.shape[0], x) #LpRf
        if self.opt['R_W_type'] in ['lin_layer', 'lin_layer_hh', 'lin_layer_hp', 'res_lin_layer', 'res_layer_hh', 'res_layer_hp']:
          RfW = torch.einsum("ij,ikj->ik", Rf, R_Ws)
        else:
          RfW = torch.matmul(Rf, R_Ws)

      f = self.alpha * LfW + (1-self.alpha) * RfW

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
      #todo check the computation of these energies
      #todo add tracking of:
      # alpha, heterophilly, diriclet and weighted dirichelt energy
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
