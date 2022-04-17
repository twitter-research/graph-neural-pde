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
from sklearn.metrics import confusion_matrix

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
    self.confusions = None

    self.val_dist_mean_feat = None
    self.val_dist_sd_feat = None
    self.test_dist_mean_feat = None
    self.test_dist_sd_feat = None
    self.val_dist_mean_label = None
    self.val_dist_sd_label = None
    self.test_dist_mean_label = None
    self.test_dist_sd_label = None

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
      self.class_dist_fig_list = []

      if opt['save_local_reports']:
        self.pdf_list = ['spectrum', 'acc_entropy', 'edge_evol', 'node_evol', 'node_scatter', 'edge_scatter']
        self.spectrum_pdf = PdfPages(f"{savefolder}/spectrum.pdf")
        self.acc_entropy_pdf = PdfPages(f"{savefolder}/acc_entropy.pdf")
        self.edge_evol_pdf = PdfPages(f"{savefolder}/edge_evol.pdf")
        self.node_evol_pdf = PdfPages(f"{savefolder}/node_evol.pdf")
        self.node_scatter_pdf = PdfPages(f"{savefolder}/node_scatter.pdf")
        self.edge_scatter_pdf = PdfPages(f"{savefolder}/edge_scatter.pdf")
        self.class_dist_pdf = PdfPages(f"{savefolder}/class_dist.pdf")

    self.epoch = 0
    self.wandb_step = 0
    self.prev_grad = None

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
      self.om_W = Parameter(torch.Tensor(in_features))
      self.om_W_eps = 0
    elif self.opt['gnl_omega'] == 'zero':
      self.om_W = torch.zeros((in_features,in_features), device=device)
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
        self.t_a = Parameter(torch.Tensor(in_features))#torch.ones(in_features))
        self.r_a = Parameter(torch.Tensor(in_features))#torch.zeros(in_features))
      elif self.opt['gnl_W_style'] == 'k_block':
        assert opt['k_blocks'] * opt['block_size'] <= in_features, 'blocks exceeded hidden dim'
        self.gnl_W_blocks = Parameter(torch.Tensor(opt['k_blocks'] * opt['block_size'], opt['block_size']))
        self.gnl_W_D = Parameter(torch.Tensor(in_features - opt['k_blocks'] * opt['block_size']))
      elif self.opt['gnl_W_style'] == 'k_diag':
        assert opt['k_diags'] % 2 == 1 and opt['k_diags'] <= in_features, 'must have odd number of k diags'
        self.gnl_W_diags = Parameter(torch.Tensor(in_features, opt['k_diags'])) #or (2k-1) * n + k * (k - 1)
      else:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))

    self.delta = Parameter(torch.Tensor([1.]))
    self.C = (data.y.max() + 1).item()  #hack!, num class for drift
    if opt['drift'] or opt['lie_trotter'] in ['gen_0','gen_1','gen_2']:
      self.drift_eps = Parameter(torch.Tensor([0.]))

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
      elif self.opt['gnl_W_style'] == 'diag_dom':
        glorot(self.W_W)
        uniform(self.t_a, a=-1, b=1)
        uniform(self.r_a, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'k_block':
        glorot(self.gnl_W_blocks)
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'k_diag':
        uniform(self.gnl_W_diags, a=-1, b=1)
      else: #sum or
        glorot(self.W_W)      # xavier_uniform_(self.W_W)

    if self.opt['gnl_measure'] in ['deg_poly', 'deg_poly_exp']:
      ones(self.m_alpha)
      ones(self.m_beta)
      ones(self.m_gamma)
    elif self.opt['gnl_measure'] in ['nodewise']:
      ones(self.measure)
    elif self.opt['gnl_measure'] in ['nodewise_exp']:
      zeros(self.measure)

    # ones(self.delta)
    # zeros(self.delta)

  def set_gnlWS(self):
    "note every W is made symetric before returning here"

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
      #todo check if I make an orthoganal matrix symetric is it still orthoganal
      # Ws = (W_hat+W_hat.T) / 2
      # return Ws

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
      #todo check if I make an orthoganal matrix symetric is it still orthoganal
      # Ws = (W_hat+W_hat.T) / 2
      # return Ws

    elif self.opt['gnl_W_style'] == 'diag_dom':
      # W_sum = self.t_a * torch.abs(self.W_W).sum(dim=1) + self.r_a
      # W_temp = torch.cat([self.W_W, W_sum.unsqueeze(-1)], dim=1)
      # W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      # Ws = (W+W.T) / 2

      W_temp = torch.cat([self.W_W, torch.zeros((self.in_features, 1), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      W = (W+W.T) / 2
      W_sum = self.t_a * torch.abs(self.W).sum(dim=1) + self.r_a
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

  def get_energy_gradient(self, x, tau, tau_transpose, attentions, edge_index, n):
    row_sum = scatter_add(attentions, edge_index[0], dim=0, dim_size=n)
    deg_inv_sqrt = torch.pow(row_sum, -0.5)
    src_x, dst_x = self.get_src_dst(x)
    src_deg_inv_sqrt, dst_deg_inv_sqrt = self.get_src_dst(deg_inv_sqrt)
    src_term = (tau * src_x * src_deg_inv_sqrt.unsqueeze(dim=-1))
    dst_term = (tau_transpose * dst_x * dst_deg_inv_sqrt.unsqueeze(dim=-1))
    energy_gradient = (src_term - dst_term) @ self.W
    return energy_gradient


  def do_diffusion(self, t):
    if self.opt['drift']:
      return True
    if self.opt['lie_trotter'] == 'gen_0':
      return True
    if self.opt['lie_trotter'] == 'gen_1':
      for rng in self.opt['diffusion_ranges']:
        if t >= rng[0] and t < rng[1]:
          return True
    if self.opt['lie_trotter'] == 'gen_2':
      if self.opt['lie_trotter_block'] == 'diffusion':
        return True
    return False

  def do_drift(self, t):
    if self.opt['drift']:
      return True
    if self.opt['lie_trotter'] == 'gen_0':
      return True
    if self.opt['lie_trotter'] == 'gen_1':
      for rng in self.opt['drift_ranges']:
        if t >= rng[0] and t < rng[1]:
          return True
    if self.opt['lie_trotter'] == 'gen_2':
      if self.opt['lie_trotter_block'] == 'drift':
        return True
    return False


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
      # dynamics_type = "diffusion"#self.dynamics_type(t)
      # if dynamics_type == "diffusion":
      if self.do_diffusion(t):
        if self.opt['gnl_measure'] == 'deg_poly':
          deg = degree(self.edge_index[0], self.n_nodes)
          measure = self.m_alpha * deg ** self.m_beta + self.m_gamma
          src_meas, dst_meas = self.get_src_dst(measure)
          measures_src_dst = 1 / (src_meas * dst_meas)
        elif self.opt['gnl_measure'] == 'nodewise':
          measure = self.measure
          src_meas, dst_meas = self.get_src_dst(measure)
          measures_src_dst = 1 / (src_meas * dst_meas)
        elif self.opt['gnl_measure'] == 'deg_poly_exp':
          deg = degree(self.edge_index[0], self.n_nodes)
          measure = torch.exp(self.m_alpha * deg ** self.m_beta + self.m_gamma)
          src_meas, dst_meas = self.get_src_dst(measure)
          measures_src_dst = 1 / (src_meas * dst_meas)
        elif self.opt['gnl_measure'] == 'nodewise_exp':
          measure = torch.exp(self.measure)
          src_meas, dst_meas = self.get_src_dst(measure)
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
            self.Omega =  (1 - 2 * self.om_W_eps) * torch.eye(self.in_features, device=self.device) - self.om_W_eps * self.om_W_attr @ self.om_W_attr.T + (1 - self.om_W_eps) * self.om_W_rep @ self.om_W_rep.T
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

          # self.gnl_W = (self.W_W + self.W_W.T) / 2 #instead set at GNN level using set_gnlWS

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

      if self.do_drift(t):
        if not self.do_diffusion(t):
          f = 0
        f = torch.exp(self.drift_eps) * f

        logits = torch.softmax(self.GNN_m2(x), dim=1)
        eye = torch.eye(self.C, device=self.device)
        dist_labels = logits.unsqueeze(-1) - eye.unsqueeze(0) #[num_nodes, c, 1] - [1, c, c]
        eta_hat = torch.sum(torch.abs(dist_labels),dim=1)  #sum abs distances for each node over features
        P = self.GNN_m2.weight
        index = list(range(self.C))
        for l in range(self.C):
          idx = index[:l] + index[l + 1:]
          q_l = eta_hat[:,l] * logits[:,l]
          eta_l = torch.prod(eta_hat[:,idx]**2, dim=1) * q_l
          f -= (-0.5 * measure.unsqueeze(-1) * torch.outer(eta_l, P[l]) + torch.outer(eta_l, torch.ones(logits.shape[1], device=self.device)) * logits @ P)/ torch.exp(self.drift_eps)


    #todo project every node onto embedding TSNE coordinate basis - https://discuss.pytorch.org/t/t-sne-for-pytorch/44264

    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.get_evol_stats:#not self.training:
      with torch.no_grad():
        #todo these energy formulas are wrong
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
        sm_logits = torch.softmax(logits, dim=1)
        homophil = homophily(edge_index=self.edge_index, y=pred)
        L2dist = torch.sqrt(torch.sum((src_x - dst_x) ** 2, dim=1))
        conf_mat, train_cm, val_cm, test_cm = self.get_confusion(self.data, pred, norm_type='true') #'all')
        eval_means_feat, eval_sds_feat = self.get_distances(self.data, x, self.C, base_mask=self.data.train_mask, eval_masks=[self.data.val_mask, self.data.test_mask])
        eval_means_label, eval_sds_label = self.get_distances(self.data, sm_logits, self.C, base_mask=self.data.train_mask, eval_masks=[self.data.val_mask, self.data.test_mask])

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
          self.confusions = [conf_mat, train_cm, val_cm, test_cm]

          self.val_dist_mean_feat = eval_means_feat[0]
          self.val_dist_sd_feat = eval_sds_feat[0]
          self.test_dist_mean_feat = eval_means_feat[1]
          self.test_dist_sd_feat = eval_sds_feat[1]

          self.val_dist_mean_label = eval_means_label[0]
          self.val_dist_sd_label = eval_sds_label[0]
          self.test_dist_mean_label = eval_means_label[1]
          self.test_dist_sd_label = eval_sds_label[1]

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

          if len(self.confusions[0].shape) == 2:
            self.confusions[0] = torch.stack((self.confusions[0], conf_mat), dim=-1)
            self.confusions[1] = torch.stack((self.confusions[1], train_cm), dim=-1)
            self.confusions[2] = torch.stack((self.confusions[2], val_cm), dim=-1)
            self.confusions[3] = torch.stack((self.confusions[3], test_cm), dim=-1)

            self.val_dist_mean_feat = torch.stack((self.val_dist_mean_feat, eval_means_feat[0]), dim=-1)
            self.val_dist_sd_feat = torch.stack((self.val_dist_sd_feat, eval_sds_feat[0]), dim=-1)
            self.test_dist_mean_feat = torch.stack((self.test_dist_mean_feat, eval_means_feat[1]), dim=-1)
            self.test_dist_sd_feat = torch.stack((self.test_dist_sd_feat, eval_sds_feat[1]), dim=-1)

            self.val_dist_mean_label = torch.stack((self.val_dist_mean_label, eval_means_label[0]), dim=-1)
            self.val_dist_sd_label = torch.stack((self.val_dist_sd_label, eval_sds_label[0]), dim=-1)
            self.test_dist_mean_label = torch.stack((self.test_dist_mean_label, eval_means_label[1]), dim=-1)
            self.test_dist_sd_label = torch.stack((self.test_dist_sd_label, eval_sds_label[1]), dim=-1)

          else:
            self.confusions[0] = torch.cat((self.confusions[0], conf_mat.unsqueeze(-1)), dim=-1)
            self.confusions[1] = torch.cat((self.confusions[1], train_cm.unsqueeze(-1)), dim=-1)
            self.confusions[2] = torch.cat((self.confusions[2], val_cm.unsqueeze(-1)), dim=-1)
            self.confusions[3] = torch.cat((self.confusions[3], test_cm.unsqueeze(-1)), dim=-1)

            self.val_dist_mean_feat = torch.cat((self.val_dist_mean_feat, eval_means_feat[0].unsqueeze(-1)),dim=-1)
            self.val_dist_sd_feat = torch.cat((self.val_dist_sd_feat, eval_sds_feat[0].unsqueeze(-1)),dim=-1)
            self.test_dist_mean_feat = torch.cat((self.test_dist_mean_feat, eval_means_feat[1].unsqueeze(-1)),dim=-1)
            self.test_dist_sd_feat = torch.cat((self.test_dist_sd_feat, eval_sds_feat[1].unsqueeze(-1)),dim=-1)

            self.val_dist_mean_label = torch.cat((self.val_dist_mean_label, eval_means_label[0].unsqueeze(-1)),dim=-1)
            self.val_dist_sd_label = torch.cat((self.val_dist_sd_label, eval_sds_label[0].unsqueeze(-1)),dim=-1)
            self.test_dist_mean_label = torch.cat((self.test_dist_mean_label, eval_means_label[1].unsqueeze(-1)),dim=-1)
            self.test_dist_sd_label = torch.cat((self.test_dist_sd_label, eval_sds_label[1].unsqueeze(-1)),dim=-1)

        ### extra values for terminal step
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
          sm_logits = torch.softmax(logits, dim=1)
          train_acc, val_acc, test_acc = test(logits, self.data)
          pred = logits.max(1)[1]
          homophil = homophily(edge_index=self.edge_index, y=pred)
          L2dist = torch.sqrt(torch.sum((src_z - dst_z) ** 2, dim=1))
          conf_mat, train_cm, val_cm, test_cm = self.get_confusion(self.data, pred, norm_type='true')  # 'all')
          eval_means_feat, eval_sds_feat = self.get_distances(self.data, z, self.C, base_mask=self.data.train_mask,
                                                              eval_masks=[self.data.val_mask, self.data.test_mask])
          eval_means_label, eval_sds_label = self.get_distances(self.data, sm_logits, self.C,
                                                                base_mask=self.data.train_mask,
                                                                eval_masks=[self.data.val_mask, self.data.test_mask])

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

          self.confusions[0] = torch.cat((self.confusions[0], conf_mat.unsqueeze(-1)), dim=-1)
          self.confusions[1] = torch.cat((self.confusions[1], train_cm.unsqueeze(-1)), dim=-1)
          self.confusions[2] = torch.cat((self.confusions[2], val_cm.unsqueeze(-1)), dim=-1)
          self.confusions[3] = torch.cat((self.confusions[3], test_cm.unsqueeze(-1)), dim=-1)

          self.val_dist_mean_feat = torch.cat((self.val_dist_mean_feat, eval_means_feat[0].unsqueeze(-1)), dim=-1)
          self.val_dist_sd_feat = torch.cat((self.val_dist_sd_feat, eval_sds_feat[0].unsqueeze(-1)), dim=-1)
          self.test_dist_mean_feat = torch.cat((self.test_dist_mean_feat, eval_means_feat[1].unsqueeze(-1)), dim=-1)
          self.test_dist_sd_feat = torch.cat((self.test_dist_sd_feat, eval_sds_feat[1].unsqueeze(-1)), dim=-1)

          self.val_dist_mean_label = torch.cat((self.val_dist_mean_label, eval_means_label[0].unsqueeze(-1)), dim=-1)
          self.val_dist_sd_label = torch.cat((self.val_dist_sd_label, eval_sds_label[0].unsqueeze(-1)), dim=-1)
          self.test_dist_mean_label = torch.cat((self.test_dist_mean_label, eval_means_label[1].unsqueeze(-1)), dim=-1)
          self.test_dist_sd_label = torch.cat((self.test_dist_sd_label, eval_sds_label[1].unsqueeze(-1)), dim=-1)

        self.wandb_step += 1

      # if self.opt['greed_momentum'] and self.prev_grad:
      #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      #   self.prev_grad = f

    return f

  def get_confusion(self, data, pred, norm_type):
    # conf_mat = confusion_matrix(data.y, pred, normalize=norm_type)
    # torch_conf_mat = self.torch_confusion(data.y, pred, norm_type)
    # print(torch.allclose(torch.from_numpy(conf_mat), torch_conf_mat, rtol=0.001))
    # train_cm = confusion_matrix(data.y[data.train_mask], pred[data.train_mask], normalize=norm_type)
    # val_cm = confusion_matrix(data.y[data.val_mask], pred[data.val_mask], normalize=norm_type)
    # test_cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask], normalize=norm_type)
    num_class = self.C
    conf_mat = self.torch_confusion(data.y, pred, num_class, norm_type)
    train_cm = self.torch_confusion(data.y[data.train_mask], pred[data.train_mask], num_class, norm_type)
    val_cm = self.torch_confusion(data.y[data.val_mask], pred[data.val_mask], num_class, norm_type)
    test_cm = self.torch_confusion(data.y[data.test_mask], pred[data.test_mask], num_class, norm_type)
    return conf_mat, train_cm, val_cm, test_cm

  def torch_confusion(self, labels, pred, num_class, norm_type):
    '''
    Truth - row i
    Pred - col j
    '''
    num_nodes = labels.shape[0]
    conf_mat = torch.zeros((num_class, num_class), dtype=torch.double, device=self.device)
    for i in range(num_class):
      for j in range(num_class):
        conf_mat[i,j] = ((labels==i).long() * (pred==j).long()).sum()
    if norm_type == None:
      pass
    elif norm_type == 'true':
      trues = torch.zeros(num_class, dtype=torch.double, device=self.device)
      for c in range(num_class):
        trues[c] = (labels == c).sum()
      conf_mat = conf_mat / trues.unsqueeze(-1)
    elif norm_type == 'pred':
      preds = torch.zeros(num_class, dtype=torch.double, device=self.device)
      for c in range(num_class):
        preds[c] = (pred == c).sum()
      conf_mat = conf_mat / preds.unsqueeze(0)
    elif norm_type == 'all':
      conf_mat / num_nodes
    return conf_mat

  def get_distances(self, data, x, num_class, base_mask, eval_masks):
    #this should work for features or preds/label space
    base_av = torch.zeros((num_class, x.shape[-1]), device=self.device)
    #calculate average hidden state per class in the baseline set - [C, d]
    for c in range(num_class):
      base_c_mask = data.y[base_mask] == c
      base_av_c = x[base_mask][base_c_mask].mean(dim=0)
      base_av[c] = base_av_c

    #for every node calcualte the L2 distance - [N, C] and [N, C]
    dist = x.unsqueeze(-1) - base_av.T.unsqueeze(0)
    L2_dist = torch.sqrt(torch.sum(dist**2, dim=1))

    #for every node in each true class in the val/test sets calc the distances away from the average train set for each class
    eval_means = []
    eval_sds = []
    for eval_mask in eval_masks:
      eval_dist_mean = torch.zeros((num_class, num_class), device=self.device)
      eval_dist_sd = torch.zeros((num_class, num_class), device=self.device)
      for c in range(num_class):
        base_c_mask = data.y[eval_mask] == c
        eval_dist_mean[c] = L2_dist[eval_mask][base_c_mask].mean(dim=0)
        eval_dist_sd[c] = L2_dist[eval_mask][base_c_mask].std(dim=0)

      eval_means.append(eval_dist_mean)
      eval_sds.append(eval_dist_sd)
    #output: rows base_class, cols eval_class
    return eval_means, eval_sds


  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
