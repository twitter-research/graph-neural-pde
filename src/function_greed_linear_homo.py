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
from utils import MaxNFEException
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      pass #not used for now but leaving here in case we split the metric like we do diffusion
      # self.output_var_x = nn.Parameter(torch.ones(1))
      # self.lengthscale_x = nn.Parameter(torch.ones(1))
      # self.output_var_p = nn.Parameter(torch.ones(1))
      # self.lengthscale_p = nn.Parameter(torch.ones(1))
      # self.Qx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      # self.init_weights(self.Qx)
      # self.Vx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      # self.init_weights(self.Vx)
      # self.Kx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      # self.init_weights(self.Kx)
      #
      # self.Qp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      # self.init_weights(self.Qp)
      # self.Vp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      # self.init_weights(self.Vp)
      # self.Kp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      # self.init_weights(self.Kp)

    else:
      self.Q = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.Q)
      self.V = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.V)
      self.K = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)
    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5) #not this https://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers

  def forward(self, x, edge):
    """
    x might be [features, augmentation, positional encoding, labels]
    """
    # if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      pass #not used for now but leaving here in case we split the metric like we do diffusion
      # label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      # p = x[:, self.opt['feat_hidden_dim']: label_index]
      # x = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)
      #
      # qx = self.Qx(x)
      # kx = self.Kx(x)
      # vx = self.Vx(x)
      # # perform linear operation and split into h heads
      # kx = kx.view(-1, self.h, self.d_k)
      # qx = qx.view(-1, self.h, self.d_k)
      # vx = vx.view(-1, self.h, self.d_k)
      # # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      # kx = kx.transpose(1, 2)
      # qx = qx.transpose(1, 2)
      # vx = vx.transpose(1, 2)
      # src_x = qx[edge[0, :], :, :]
      # dst_x = kx[edge[1, :], :, :]
      #
      # qp = self.Qp(p)
      # kp = self.Kp(p)
      # vp = self.Vp(p)
      # # perform linear operation and split into h heads
      # kp = kp.view(-1, self.h, self.d_k)
      # qp = qp.view(-1, self.h, self.d_k)
      # vp = vp.view(-1, self.h, self.d_k)
      # # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      # kp = kp.transpose(1, 2)
      # qp = qp.transpose(1, 2)
      # vp = vp.transpose(1, 2)
      # src_p = qp[edge[0, :], :, :]
      # dst_p = kp[edge[1, :], :, :]
      #
      # prods = self.output_var_x ** 2 * torch.exp(
      #   -torch.sum((src_x - dst_x) ** 2, dim=1) / (2 * self.lengthscale_x ** 2)) \
      #         * self.output_var_p ** 2 * torch.exp(
      #   -torch.sum((src_p - dst_p) ** 2, dim=1) / (2 * self.lengthscale_p ** 2))
      #
      # v = None

    else:
      q = self.Q(x)
      k = self.K(x)
      v = self.V(x)

      # perform linear operation and split into h heads

      k = k.view(-1, self.h, self.d_k)
      q = q.view(-1, self.h, self.d_k)
      v = v.view(-1, self.h, self.d_k)

      # transpose to get dimensions [n_nodes, attention_dim, n_heads]

      k = k.transpose(1, 2)
      q = q.transpose(1, 2)
      v = v.transpose(1, 2)

      src_q = q[edge[0, :], :, :]
      dst_k = k[edge[1, :], :, :]
      prods1 = torch.sum(src_q * dst_k, dim=1) / np.sqrt(self.d_k)
      if self.opt['test_grand_metric']:
        attention = softmax(prods1, edge[self.opt['attention_norm_idx']])
        return attention, (None, None)
      else:
        src_k = k[edge[0, :], :, :]
        dst_q = q[edge[1, :], :, :]
        prods2 = torch.sum(src_k * dst_q, dim=1) / np.sqrt(self.d_k)
        attention = (softmax(prods1, edge[self.opt['attention_norm_idx']]) + softmax(prods2, edge[self.opt['attention_norm_idx']])) / 2
        return attention, (None, None)

    # if self.opt['attention_type'] == "scaled_dot":
    #   prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
    #
    # elif self.opt['attention_type'] == "cosine_sim":
    #   cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
    #   prods = cos(src, dst_k)
    #
    # elif self.opt['attention_type'] == "pearson":
    #   src_mu = torch.mean(src, dim=1, keepdim=True)
    #   dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
    #   src = src - src_mu
    #   dst_k = dst_k - dst_mu
    #   cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
    #   prods = cos(src, dst_k)

    # if self.opt['reweight_attention'] and self.edge_weights is not None:
    #   prods = prods * self.edge_weights.unsqueeze(dim=1)
    # if self.opt['square_plus']:
    #   attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
    # else:
    #   attention = softmax(prods, edge[self.opt['attention_norm_idx']])
    # return attention, (v, prods) #attention here is size


class ODEFuncGreedLinH(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedLinH, self).__init__(in_features, out_features, opt, data, device, bias=False)

    self.energy = 0
    self.epoch = 0
    self.wandb_step = 0
    self.prev_grad = None
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

    self.reset_linH_parameters()

    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt, #check out_features is attention_dim
                                                          device, edge_weights=self.edge_weight).to(device)


  def reset_linH_parameters(self): #todo rename the param reset() functions in scaledDP and greed_linear aswell
    if not self.opt['test_tau_symmetric']:
      glorot(self.Qx)
      glorot(self.Qp)
    glorot(self.Kx)
    glorot(self.Kp)
    zeros(self.bias)
    glorot(self.W)

  def set_x_0(self, x_0):
    self.x_0 = x_0.clone().detach()
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
    if self.opt['test_tau_symmetric']:
      self.tau_f_0, self.tau_f_transpose_0 = self.get_tau(self.xf_0, self.Kx, self.Kx)
      self.tau_p_0, self.tau_p_transpose_0 = self.get_tau(self.p_0, self.Kp, self.Kp)
    else:
      self.tau_f_0, self.tau_f_transpose_0 = self.get_tau(self.xf_0, self.Qx, self.Kx)
      self.tau_p_0, self.tau_p_transpose_0 = self.get_tau(self.p_0, self.Qp, self.Kp)

  def set_L0(self):
    #here get metric is the symetric attention matrix
    # tau_0, tau_transpose_0 = self.get_tau(self.x_0)
    # metric_0 = self.get_metric(x_0, tau_0, tau_transpose_0)

    attention, _ = self.multihead_att_layer(self.x_0, self.edge_index)
    self.mean_attention_0 = attention.mean(dim=1)

    if self.opt['test_omit_metric']:
      gamma = torch.ones(self.mean_attention_0.shape, device=self.mean_attention_0.device) #setting metric equal to adjacency
    else:
      gamma = self.mean_attention_0

    self.Lf_0 = self.get_laplacian_linear(gamma, self.tau_f_0, self.tau_f_transpose_0)
    self.Lp_0 = self.get_laplacian_linear(gamma, self.tau_p_0, self.tau_p_transpose_0)


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

    L = self.get_laplacian_form(T1, T0)
    return L


  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    Ws = self.W @ self.W.t()  # output a [d,d] tensor

    # x is [(features, pos_encs) * aug_factor, lables] but it's safe to assume aug_factor == 1
    label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
    p = x[:, self.opt['feat_hidden_dim']: label_index]
    xf = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)

    # with torch.no_grad(): #these things only go into calculating Energy not forward
    #   metric_0 = self.get_metric(self.x_0, self.tau_0, self.tau_transpose_0)
    #   _, eta = self.get_gamma(metric_0, self.opt['gamma_epsilon'])
    #   tau, tau_transpose = self.tau_0, self.tau_transpose_0 #assigning for energy calcs

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)

    ##PURE GREED
    ff = torch_sparse.spmm(edges, -self.Lf_0, xf.shape[0], xf.shape[0], xf)
    ff = torch.matmul(ff, Ws)
    ff = ff - self.mu * (xf - self.xf_0)
    fp = torch_sparse.spmm(edges, -self.Lp_0, p.shape[0], p.shape[0], p)
    f = torch.cat([ff, fp], dim=1) #assuming don't have any augmentation or labels

    ###mix
    # if not self.opt['no_alpha_sigmoid']:
    #   alpha = torch.sigmoid(self.alpha_train)
    # else:
    #   alpha = self.alpha_train
    # ff = alpha * torch_sparse.spmm(edges, -self.Lf_0, xf.shape[0], xf.shape[0], xf)
    # ff = torch.matmul(ff, Ws)
    # ff = ff  ###newline
    # # ff = ff - self.mu * (ff - self.xf_0) #check xf0
    # ff = ff + self.mu * (self.xf_0) ###newline
    # fp = alpha * torch_sparse.spmm(edges, -self.Lp_0, p.shape[0], p.shape[0], p)
    # f = torch.cat([ff, fp], dim=1) #assuming don't have any augmentation or labels

    ###FUNCTION LAPLACIAN
    # ax = self.sparse_multiply(x)
    # if not self.opt['no_alpha_sigmoid']:
    #   alpha = torch.sigmoid(self.alpha_train)
    # else:
    #   alpha = self.alpha_train
    # f = alpha * (ax - x)
    # if self.opt['add_source']:
    #   f = f + self.beta_train * self.x0

    # if self.opt['test_omit_metric'] and self.opt['test_mu_0']: #energy to use when Gamma is -adjacency and not the pullback and mu == 0
    #   energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2)
    # elif self.opt['test_omit_metric']: #energy to use when Gamma is -adjacency and not the pullback and mu != 0
    #   energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2) + self.mu * torch.sum((x - self.x0) ** 2)
    # else:
    #   energy = self.get_energy(x, eta)

    R1 = 0
    R2 = 0
    energy = 0
    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.training:
      wandb.log({f"gf_e{self.epoch}_energy_change": energy - self.energy, f"gf_e{self.epoch}_energy": energy,
                 f"gf_e{self.epoch}_f": f ** 2, f"gf_e{self.epoch}_L": torch.sum(L ** 2),
                 f"gf_e{self.epoch}_R1": torch.sum(R1 ** 2), f"gf_e{self.epoch}_R2": torch.sum(R2 ** 2), f"gf_e{self.epoch}_mu": self.mu,
                 "grad_flow_step": self.wandb_step})
      self.wandb_step += 1

    self.energy = energy

    # if self.opt['greed_momentum'] and self.prev_grad:
    #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
    #   self.prev_grad = f
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
