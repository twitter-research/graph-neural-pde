"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
from torch.nn.init import uniform
import numpy as np
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import degree, softmax
from torch_sparse import coalesce, transpose
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from torch.nn import Parameter
import wandb
from function_greed import ODEFuncGreed
from utils import MaxNFEException, sym_row_col, sym_row_col_att, sym_row_col_att_measure, gram_schmidt, sym_row_col_att_relaxed, sigmoid_deriv, tanh_deriv, squareplus_deriv

from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer
from function_transformer_attention_greed import SpGraphTransAttentionLayer_greed


@torch.no_grad()
# def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
def test(logits, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  # model.eval()
  # feat = data.x
  # if model.opt['use_labels']:
  #   feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  # logits, accs = model(feat, pos_encoding), []
  accs = []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs

class ODEFuncGreedNonLin(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedNonLin, self).__init__(in_features, out_features, opt, data, device, bias=False)

    self.data = data
    self.energy = 0
    self.fOmf = None
    self.attentions = None
    self.L2dist = None
    self.train_accs = None
    self.val_accs = None
    self.test_accs = None

    self.epoch = 0
    self.wandb_step = 0
    self.prev_grad = None

    if self.opt['gnl_omega'] == 'sum':
      # self.om_W = -torch.eye(in_features, in_features)/2
      # self.om_W = Parameter(-torch.eye(in_features, in_features)/2)
      self.om_W = Parameter(torch.Tensor(in_features, in_features))
      self.om_W_eps = 0
    elif self.opt['gnl_omega'] == 'product':
      self.om_W = Parameter(torch.Tensor(in_features, opt['dim_p_w']))

    elif self.opt['gnl_omega'] == 'attr_rep':
      self.om_W_attr = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
      self.om_W_rep = Parameter(torch.Tensor(in_features, opt['dim_p_w']))
      self.om_W_eps = Parameter(torch.Tensor([0.85]))
      # self.om_W_eps = torch.Tensor([1.0])
      self.om_W_nu = torch.Tensor([0.1])

    elif self.opt['gnl_omega'] == 'diag':
      self.om_W = Parameter(torch.Tensor(in_features))
      self.om_W_eps = 0


    if opt['gnl_measure'] == 'deg_poly':
      self.m_alpha = Parameter(torch.Tensor([1.]))
      self.m_beta = Parameter(torch.Tensor([1.]))
      self.m_gamma = Parameter(torch.Tensor([0.]))
    elif opt['gnl_measure'] == 'nodewise':
      self.measure = Parameter(torch.Tensor(self.n_nodes))
    elif opt['gnl_measure'] == 'ones':
      pass

    if self.opt['gnl_style'] == 'softmax_attention':
      self.multihead_att_layer = SpGraphTransAttentionLayer_greed(in_features, out_features, opt,
                                                                  # check out_features is attention_dim
                                                                  device, edge_weights=self.edge_weight).to(device)

    self.delta = Parameter(torch.Tensor([1.]))
    # self.delta = torch.Tensor([2.0])

    self.reset_nonlinG_parameters()

  def reset_nonlinG_parameters(self):
    if self.opt['gnl_omega'] == 'sum':
      glorot(self.om_W)
    elif self.opt['gnl_omega'] == 'product':
      glorot(self.om_W)
    elif self.opt['gnl_omega'] == 'attr_rep':
      glorot(self.om_W_attr)
      glorot(self.om_W_rep)
      # # zeros(self.om_W_attr)
      # zeros(self.om_W_rep)
      # constant(self.om_W_attr, 0.0001)
      # constant(self.om_W_rep, 0.0001)
    elif self.opt['gnl_omega'] == 'diag':
      # zeros(self.om_W)
      uniform(self.om_W, a=-1, b=1)

    if self.opt['gnl_measure'] == 'deg_poly':
      ones(self.m_alpha)
      ones(self.m_beta)
      ones(self.m_gamma)
    elif self.opt['gnl_measure'] == 'nodewise':
      ones(self.measure)
    elif self.opt['gnl_measure'] == 'ones':
      pass

    # ones(self.delta)
    # zeros(self.delta)

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

    if self.opt['beltrami']:
      # Ws = self.Ws
      # # x is [(features, pos_encs) * aug_factor, lables] but here assume aug_factor == 1
      # label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      # p = x[:, self.opt['feat_hidden_dim']: label_index]
      # xf = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)
      # ff = torch_sparse.spmm(edges, -self.Lf_0, xf.shape[0], xf.shape[0], xf)
      # ff = torch.matmul(ff, Ws)
      # ff = ff - self.mu * (xf - self.xf_0)
      # fp = torch_sparse.spmm(edges, -self.Lp_0, p.shape[0], p.shape[0], p)
      # f = torch.cat([ff, fp], dim=1)
      pass
    else:

      if self.opt['gnl_measure'] == 'deg_poly':
        deg = degree(self.edge_index[0], self.n_nodes)
        measure = self.m_alpha * deg ** self.m_beta + self.m_gamma
        src_meas, dst_meas = self.get_src_dst(measure)
        measures_src_dst = 1 / (src_meas * dst_meas)
      elif self.opt['gnl_measure'] == 'nodewise':
        src_meas, dst_meas = self.get_src_dst(self.measure)
        measures_src_dst = 1 / (src_meas * dst_meas)
      elif self.opt['gnl_measure'] == 'ones':
        src_meas = 1
        dst_meas = 1
        measures_src_dst = 1

      #scaled-dot method
      if self.opt['gnl_style'] == 'scaled_dot':
        if self.opt['gnl_omega'] == 'sum':
          self.Omega = self.om_W + self.om_W.T
        elif self.opt['gnl_omega'] == 'product':
          self.Omega = self.om_W @ self.om_W.T
        elif self.opt['gnl_omega'] == 'attr_rep':
          # Omega = self.om_W_nu * (1 - 2 * self.om_W_eps) - self.om_W_eps * self.om_W_attr @ self.om_W_attr.T + (1 - self.om_W_eps) * self.om_W_rep @ self.om_W_rep.T
          self.Omega =  (1 - 2 * self.om_W_eps) * torch.eye(self.in_features) - self.om_W_eps * self.om_W_attr @ self.om_W_attr.T + (1 - self.om_W_eps) * self.om_W_rep @ self.om_W_rep.T
        elif self.opt['gnl_omega'] == 'diag':
          self.Omega = torch.diag(self.om_W)


        if self.opt['gnl_omega_norm'] == 'tanh':
          self.Omega = torch.tanh(self.Omega)
        elif self.opt['gnl_omega_norm'] == 'rowSum':
          D = self.Omega.abs().sum(dim=1)
          self.Omega = torch.diag(torch.pow(D, -0.5)) @ self.Omega @ torch.diag(torch.pow(D, -0.5))
        else:
          pass

        src_x, dst_x = self.get_src_dst(x)
        # fOmf = torch.einsum("ij,jj,ij->i", src_x, self.Omega, dst_x)
        fOmf = torch.einsum("ij,jk,ik->i", src_x, self.Omega, dst_x)

        if self.opt['gnl_activation'] == "sigmoid_deriv":
          attention = sigmoid_deriv(fOmf)
        elif self.opt['gnl_activation'] == "tanh_deriv":
          attention = tanh_deriv(fOmf)
        elif self.opt['gnl_activation'] == "squareplus_deriv":
          attention = squareplus_deriv(fOmf)
        elif self.opt['gnl_activation'] == "exponential":
          attention = torch.exp(fOmf)
        else:
          attention = fOmf

        MThM = measures_src_dst * attention
        f = torch_sparse.spmm(self.edge_index, -MThM, x.shape[0], x.shape[0], x @ self.Omega)

      #softmax_attention method
      elif self.opt['gnl_style'] == 'softmax_attention':
        attention_h, _ = self.multihead_att_layer(x, self.edge_index)
        attention = attention_h.mean(dim=1)

        self.Omega = self.multihead_att_layer.QK.weight.T @ self.multihead_att_layer.QK.weight

        f1 = torch_sparse.spmm(self.edge_index, -attention / src_meas, x.shape[0], x.shape[0], x @ self.Omega)
        index_t, att_t = transpose(self.edge_index, attention, x.shape[0], x.shape[0])
        f2 = torch_sparse.spmm(index_t, -att_t / dst_meas, x.shape[0], x.shape[0], x @ self.Omega)
        f = f1 + f2

    f = f - self.delta * x #break point np.isnan(f.sum().detach().numpy())

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


    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and not self.training:
      with torch.no_grad():
        if self.opt['gnl_activation'] == "sigmoid_deriv":
          energy = torch.sum(torch.sigmoid(fOmf))
        elif self.opt['gnl_activation'] == "squareplus_deriv":
          energy = torch.sum((fOmf + torch.sqrt(fOmf ** 2 + 4)) / 2)
        elif self.opt['gnl_activation'] == "exponential":
          energy = torch.sum(torch.exp(fOmf))

        energy = energy + 0.5 * self.delta * torch.sum(x**2)

        if self.opt['test_mu_0'] and self.opt['add_source']:
          energy = energy - self.beta_train * torch.sum(x * self.x0)
        elif not self.opt['test_mu_0']:
          energy = energy + self.mu * torch.sum((x - self.x0) ** 2)
        else:
          energy = 0
          self.energy = energy

        wandb.log({f"gf_e{self.epoch}_energy_change": energy - self.energy, f"gf_e{self.epoch}_energy": energy,
                   f"gf_e{self.epoch}_f": (f**2).sum(),
                   f"gf_e{self.epoch}_x": (x ** 2).sum(),
                   "grad_flow_step": self.wandb_step})


        logits = self.GNN_m2(x)
        train_acc, val_acc, test_acc = test(logits, self.data)

        L2dist = torch.sqrt(torch.sum((src_x - dst_x) ** 2, dim=1))
        if self.attentions is None:
          self.attentions = attention.unsqueeze(0)
          self.fOmf = fOmf.unsqueeze(0)
          self.L2dist = L2dist.unsqueeze(0)
          self.train_accs = [train_acc]
          self.val_accs = [val_acc]
          self.test_accs = [test_acc]

        else:
          self.attentions = torch.cat([self.attentions, attention.unsqueeze(0)], dim=0)
          self.fOmf = torch.cat([self.fOmf, fOmf.unsqueeze(0)], dim=0)
          self.L2dist = torch.cat([self.L2dist, L2dist.unsqueeze(0)], dim=0)
          self.train_accs.append(train_acc)
          self.val_accs.append(val_acc)
          self.test_accs.append(test_acc)

        self.wandb_step += 1

      # if self.opt['greed_momentum'] and self.prev_grad:
      #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      #   self.prev_grad = f

    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
