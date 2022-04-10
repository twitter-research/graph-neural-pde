"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""
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
from torch_geometric.utils import degree, softmax, homophily
from torch_sparse import coalesce, transpose
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from torch_scatter import scatter_mean
from torch.nn import Parameter, Softmax, Softplus
from torch.distributions import Categorical
import wandb
from matplotlib.backends.backend_pdf import PdfPages

from function_greed import ODEFuncGreed
from utils import MaxNFEException, sym_row_col, sym_row_col_att, sym_row_col_att_measure, gram_schmidt, sym_row_col_att_relaxed, sigmoid_deriv, tanh_deriv, squareplus_deriv
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer
from function_transformer_attention_greed import SpGraphTransAttentionLayer_greed


@torch.no_grad()
def test(logits, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  accs = []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs

@torch.no_grad()
def get_entropies(logits, data, activation="softmax", pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  entropies_dic = {} #[]
  # https://discuss.pytorch.org/t/difficulty-understanding-entropy-in-pytorch/51014
  # https://pytorch.org/docs/stable/distributions.html
  if activation == "softmax":
    S = Softmax(dim=1)
  elif activation == "squaremax":
    S = Softplus(dim=1)

  for mask_name, mask in data('train_mask', 'val_mask', 'test_mask'):
    p_matrix = S(logits[mask])
    pred = logits[mask].max(1)[1]
    labels = data.y[mask]
    correct = pred == labels
    entropy2 = Categorical(probs=p_matrix).entropy()
    entropies_dic[f"entropy_{mask_name}_correct"] = correct.unsqueeze(0)
    entropies_dic[f"entropy_{mask_name}"] = entropy2.unsqueeze(0)

  return entropies_dic


class ODEFuncGreedNonLin(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedNonLin, self).__init__(in_features, out_features, opt, data, device, bias=False)

    self.data = data
    self.get_evol_stats = False
    self.energy = 0
    self.fOmf = None
    self.attentions = None
    self.L2dist = None
    self.node_magnitudes = None
    self.node_measures = None

    self.train_accs = None
    self.val_accs = None
    self.test_accs = None
    self.homophils = None
    self.entropies = None

    self.graph_edge_homophily = homophily(edge_index=self.edge_index, y=data.y, method='edge')
    self.graph_node_homophily = homophily(edge_index=self.edge_index, y=data.y, method='node')
    self.labels = data.y

    row, col = data.edge_index
    edge_homophils = torch.zeros(row.size(0), device=row.device)
    edge_homophils[data.y[row] == data.y[col]] = 1.
    node_homophils = scatter_mean(edge_homophils, col, 0, dim_size=data.y.size(0))
    self.edge_homophils = edge_homophils
    self.node_homophils = node_homophils
    self.degree = degree(self.edge_index[0], self.n_nodes)

    if self.opt['wandb_track_grad_flow']:
      savefolder = f"./plots/{opt['gnl_savefolder']}"
      try:
        os.mkdir(savefolder)
      except OSError:
        if os.path.exists(savefolder):
          shutil.rmtree(savefolder)
          os.mkdir(savefolder)
          print("%s exists, clearing existing images" % savefolder)
        else:
          print("Creation of the directory %s failed" % savefolder)
      else:
        print("Successfully created the directory %s " % savefolder)

      self.spectrum_fig_list = []
      self.acc_entropy_fig_list = []
      self.edge_evol_fig_list = []
      self.node_evol_fig_list = []
      self.node_scatter_fig_list = []
      self.edge_scatter_fig_list = []

      self.spectrum_pdf = PdfPages(f"{savefolder}/spectrum.pdf")
      self.acc_entropy_pdf = PdfPages(f"{savefolder}/acc_entropy.pdf")
      self.edge_evol_pdf = PdfPages(f"{savefolder}/edge_evol.pdf")
      self.node_evol_pdf = PdfPages(f"{savefolder}/node_evol.pdf")
      self.node_scatter_pdf = PdfPages(f"{savefolder}/node_scatter.pdf")
      self.edge_scatter_pdf = PdfPages(f"{savefolder}/edge_scatter.pdf")

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
    elif self.opt['gnl_omega'] == 'zero':
      self.om_W = torch.zeros((in_features,in_features), device=device)
      self.om_W_eps = 0

    if opt['gnl_measure'] == 'deg_poly':
      self.m_alpha = Parameter(torch.Tensor([1.]))
      self.m_beta = Parameter(torch.Tensor([1.]))
      self.m_gamma = Parameter(torch.Tensor([0.]))
    elif opt['gnl_measure'] == 'nodewise':
      self.measure = Parameter(torch.Tensor(self.n_nodes))
    elif opt['gnl_measure'] == 'ones':
      pass

    #'gnl_style' in 'scaled_dot' / 'softmax_attention' / 'general_graph'

    if self.opt['gnl_style'] == 'softmax_attention':
      self.multihead_att_layer = SpGraphTransAttentionLayer_greed(in_features, out_features, opt,
                                                                  # check out_features is attention_dim
                                                                  device, edge_weights=self.edge_weight).to(device)

    if self.opt['gnl_style'] == 'general_graph':
      # gnl_omega -> "gnl_W"
      if self.opt['gnl_W_style'] == 'GS':
        self.gnl_W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'cgnn':
        self.gnl_W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'diag_dom':
        self.W_W = Parameter(torch.Tensor(in_features, in_features - 1))
        self.t_a = Parameter(torch.ones(in_features))
        self.t_b = Parameter(torch.zeros(in_features))
      else:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))
        self.W_W_eps = 0 #what's this?


    self.delta = Parameter(torch.Tensor([1.]))
    # self.delta = torch.Tensor([2.0])
    if opt['drift']:
      self.C = (data.y.max()+1).item() #num class for drift
      self.drift_eps = Parameter(torch.Tensor([1.])) #placeholder for decoder

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

    if self.opt['gnl_style'] == 'general_graph':
      if self.opt['gnl_W_style'] == 'GS':
        glorot(self.gnl_W_U)
        # self.gnl_W_D
      elif self.opt['gnl_W_style'] == 'cgnn':
        glorot(self.gnl_W_U)
        # self.gnl_W_D
      else: #sum or diag_dom
        glorot(self.W_W)      # xavier_uniform_(self.W_W)

    if self.opt['gnl_measure'] == 'deg_poly':
      ones(self.m_alpha)
      ones(self.m_beta)
      ones(self.m_gamma)
    elif self.opt['gnl_measure'] == 'nodewise':
      ones(self.measure)

    # ones(self.delta)
    # zeros(self.delta)

  def set_gnlWS(self):
    if self.opt['gnl_W_style'] in ['prod']:
      return self.W_W @ self.W_W.t()  # output a [d,d] tensor
    if self.opt['gnl_W_style'] in ['sum']:
      return (self.W_W + self.W_W.t()) / 2
    # elif self.opt['W_type'] == 'diag':
    #   return torch.diag(self.W)
    # elif self.opt['W_type'] == 'residual_prod':
    #   return torch.eye(self.W.shape[0], device=x.device) + self.W @ self.W.t()  # output a [d,d] tensor

    elif self.opt['gnl_W_style'] == 'GS':#'residual_GS':
      V_hat = gram_schmidt(self.gnl_W_U)
      # W_D = torch.clamp(self.gnl_W_D, min=-1, max=1)
      W_D = torch.tanh(self.gnl_W_D)
      W_hat = V_hat @ torch.diag(W_D) @ V_hat.t()
      return W_hat
      # W_hat = V_hat @ torch.diag(torch.exp(self.gnl_W_D) - 1.5) @ V_hat.t()
      # return torch.eye(self.gnl_W.shape[0], device=self.device) + W_hat

    elif self.opt['gnl_W_style'] == 'cgnn':
      # # https://github.com/JRowbottomGit/ContinuousGNN/blob/85d47b0748a19e06e305c21e99e1dd03d36ad314/src/trainer.py
      beta = self.opt['W_beta']
      with torch.no_grad():
        W_U = self.gnl_W_U.clone()
        W_U = self.gnl_W_U.copy_((1 + beta) * W_U - beta * W_U @ W_U.t() @ W_U)

      # W_D = torch.clamp(self.gnl_W_D, min=-1, max=1) #self.gnl_W_D
      W_D = torch.tanh(self.gnl_W_D) #self.gnl_W_D

      W_hat = W_U @ torch.diag(W_D) @ W_U.t()
      return W_hat

    elif self.opt['gnl_W_style'] == 'diag_dom':
      # W_sum = self.t_a * self.W_W.sum(dim=1) + self.t_b
      W_sum = torch.zeros(self.in_features)
      W_temp = torch.cat([self.W_W, W_sum.unsqueeze(-1)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      return W

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
        measure = self.measure
        src_meas, dst_meas = self.get_src_dst(self.measure)
        measures_src_dst = 1 / (src_meas * dst_meas)
      elif self.opt['gnl_measure'] == 'ones':
        measure = torch.ones(x.shape[0], device=self.device)
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

        #method for normalising Omega to control the eigen values
        if self.opt['gnl_omega_norm'] == 'tanh':
          self.Omega = torch.tanh(self.Omega)
        elif self.opt['gnl_omega_norm'] == 'rowSum':
          D = self.Omega.abs().sum(dim=1)
          self.Omega = torch.diag(torch.pow(D, -0.5)) @ self.Omega @ torch.diag(torch.pow(D, -0.5))
        else:
          pass

        src_x, dst_x = self.get_src_dst(x)
        # fOmf = torch.einsum("ij,jj,ij->i", src_x, self.Omega, dst_x) incorrect
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

        if self.opt['gnl_omega'] == 'zero':
          self.Omega = self.om_W
        elif self.opt['gnl_omega'] == 'diag':
          self.Omega = torch.diag(self.om_W)
        else:
          self.Omega = (self.om_W + self.om_W.T) / 2

        # self.gnl_W = (self.W_W + self.W_W.T) / 2 #set at GNN level

        #get degrees
        src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)

        #calc bilinear form
        src_x, dst_x = self.get_src_dst(x)
        if not self.opt['gnl_activation'] == 'identity':
          fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W, dst_x * dst_deginvsqrt.unsqueeze(dim=1))
          #in the overleaf this is actually fWf just keeping for code homogeniety
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
            # fOmf = torch.ones(src_deginvsqrt.shape, device=self.device)
            fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                       dst_x * dst_deginvsqrt.unsqueeze(dim=1))
          attention = torch.ones(src_deginvsqrt.shape, device=self.device)

        P = attention * src_deginvsqrt * dst_deginvsqrt
        xW = x @ self.gnl_W
        f1 = torch_sparse.spmm(self.edge_index, P / src_meas, x.shape[0], x.shape[0], xW) / 2
        f2 = torch_sparse.spmm(self.edge_index, P / dst_meas, x.shape[0], x.shape[0], xW) / 2
        f = f1 + f2

        f = f - torch.diag(1 / measure) @ x @ self.Omega

    if self.opt['test_mu_0']:
      if self.opt['add_source']:
        f = f + self.beta_train * self.x0
    else:
      f = f - 0.5 * self.mu * (x - self.x0)
      # f = f - 0.5 * self.beta_train * (x - self.x0) #replacing beta with mu

    if self.opt['drift']:
      f = torch.exp(self.drift_eps) * f
      #old style found in greed linear hetero
      # drift = -self.C * f
      # for c in self.attractors.values():
      #   drift += c
      # f = f + drift

      logits = torch.softmax(self.GNN_m2(x), dim=1)
      eye = torch.eye(logits.shape[1], device=self.device)
      d = logits.unsqueeze(-1) - eye.unsqueeze(0) #[num_nodes, d, 1] - [1, d, d]
      eta_hat = torch.sum(torch.abs(d),dim=1)  #sum abs distances for each node over features
      P = self.GNN_m2.weight
      index = list(range(self.C))
      for l in range(self.C):
        idx = index[:l] + index[l + 1:]
        q_l = eta_hat[:,l] * logits[:,l]
        eta_l = torch.prod(eta_hat[:,idx]**2, dim=1) * q_l
        f -= (-0.5 * torch.outer(eta_l, P[l]) + torch.outer(eta_l, torch.ones(logits.shape[1], device=self.device)) * logits @ P)\
             / torch.exp(self.drift_eps)


    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.get_evol_stats:#not self.training:
      with torch.no_grad():
        #todo these energy formulas need checking
        if self.opt['gnl_style'] == 'scaled_dot':
          if self.opt['gnl_activation'] == "sigmoid_deriv":
            energy = torch.sum(torch.sigmoid(fOmf))
          elif self.opt['gnl_activation'] == "squareplus_deriv":
            energy = torch.sum((fOmf + torch.sqrt(fOmf ** 2 + 4)) / 2)
          elif self.opt['gnl_activation'] == "exponential":
            energy = torch.sum(torch.exp(fOmf))
          elif self.opt['gnl_activation'] == "identity":
            energy = fOmf ** 2 / 2
        else:
          energy = 0

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
        #note we could include some of the below stats in the wandb logging

        z = x
        # Activation.
        if not self.opt['XN_no_activation']:
          z = F.relu(z)
        if self.opt['fc_out']:
          z = self.fc(z)
          z = F.relu(z)
        logits = self.GNN_m2(z)
        train_acc, val_acc, test_acc = test(logits, self.data)
        pred = logits.max(1)[1]
        homophil = homophily(edge_index=self.edge_index, y=pred)
        L2dist = torch.sqrt(torch.sum((src_x - dst_x) ** 2, dim=1))

        if self.attentions is None:
          self.attentions = attention.unsqueeze(0)
          self.fOmf = fOmf.unsqueeze(0)
          self.L2dist = L2dist.unsqueeze(0)
          self.node_magnitudes = torch.sqrt(torch.sum(x**2,dim=1)).unsqueeze(0)
          self.node_measures = measure.detach().unsqueeze(0)
          self.train_accs = [train_acc]
          self.val_accs = [val_acc]
          self.test_accs = [test_acc]
          self.homophils = [homophil]
          self.entropies = get_entropies(logits, self.data)
        else:
          self.attentions = torch.cat([self.attentions, attention.unsqueeze(0)], dim=0)
          self.fOmf = torch.cat([self.fOmf, fOmf.unsqueeze(0)], dim=0)
          self.L2dist = torch.cat([self.L2dist, L2dist.unsqueeze(0)], dim=0)
          self.node_magnitudes = torch.cat([self.node_magnitudes, torch.sqrt(torch.sum(x**2,dim=1)).unsqueeze(0)], dim=0)
          self.node_measures = torch.cat([self.node_measures, measure.detach().unsqueeze(0)], dim=0)
          self.train_accs.append(train_acc)
          self.val_accs.append(val_acc)
          self.test_accs.append(test_acc)
          self.homophils.append(homophil)

          temp_entropies = get_entropies(logits, self.data)
          for key, value, in self.entropies.items():
            self.entropies[key] = torch.cat([value, temp_entropies[key]], dim=0)

        #extra values for terminal step
        if t == self.opt['time'] - self.opt['step_size']:
          z = x + self.opt['step_size'] * f #take an euler step

          src_z, dst_z = self.get_src_dst(z)
          if not self.opt['gnl_activation'] == 'identity':
            fOmf = torch.einsum("ij,jk,ik->i", src_z * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                                dst_z * dst_deginvsqrt.unsqueeze(dim=1))
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
              # fOmf = torch.ones(src_deginvsqrt.shape, device=self.device)
              fOmf = torch.einsum("ij,jk,ik->i", src_z * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                                  dst_z * dst_deginvsqrt.unsqueeze(dim=1))
            attention = torch.ones(src_deginvsqrt.shape, device=self.device)

          # Activation.
          if not self.opt['XN_no_activation']:
            z = F.relu(z)
          if self.opt['fc_out']:
            z = self.fc(z)
            z = F.relu(z)
          logits = self.GNN_m2(z)
          train_acc, val_acc, test_acc = test(logits, self.data)
          pred = logits.max(1)[1]
          homophil = homophily(edge_index=self.edge_index, y=pred)
          L2dist = torch.sqrt(torch.sum((src_z - dst_z) ** 2, dim=1))

          self.attentions = torch.cat([self.attentions, attention.unsqueeze(0)], dim=0)
          self.fOmf = torch.cat([self.fOmf, fOmf.unsqueeze(0)], dim=0)
          self.L2dist = torch.cat([self.L2dist, L2dist.unsqueeze(0)], dim=0)
          self.node_magnitudes = torch.cat([self.node_magnitudes, torch.sqrt(torch.sum(z ** 2, dim=1)).unsqueeze(0)],
                                           dim=0)
          self.node_measures = torch.cat([self.node_measures, measure.detach().unsqueeze(0)], dim=0)
          self.train_accs.append(train_acc)
          self.val_accs.append(val_acc)
          self.test_accs.append(test_acc)
          self.homophils.append(homophil)
          temp_entropies = get_entropies(logits, self.data)
          for key, value, in self.entropies.items():
            self.entropies[key] = torch.cat([value, temp_entropies[key]], dim=0)

        self.wandb_step += 1

      # if self.opt['greed_momentum'] and self.prev_grad:
      #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      #   self.prev_grad = f

    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
