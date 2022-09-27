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
from torch_geometric.utils import degree, softmax, homophily, contains_self_loops, to_dense_adj, to_undirected
from torch_sparse import coalesce, transpose
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_mean
from torch.nn import Parameter, Softmax, Softplus
from torch.distributions import Categorical
import wandb
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix

from function_greed import ODEFuncGreed
from utils import MaxNFEException, sym_row_col, sym_row_col_att, sym_row_col_att_measure, gram_schmidt, sym_row_col_att_relaxed, sigmoid_deriv, tanh_deriv, squareplus_deriv, make_symmetric_unordered
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer
from function_transformer_attention_greed import SpGraphTransAttentionLayer_greed
from greed_reporting_fcts import set_reporting_attributes, set_folders_pdfs, generate_stats, append_stats, stack_stats
from base_classes import MLP

class ODEFuncGreedNonLin(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedNonLin, self).__init__(in_features, out_features, opt, data, device, bias=False)
    self.data = data

    if self.opt['wandb_track_grad_flow']:
      set_reporting_attributes(self, data, opt)
      set_folders_pdfs(self, opt)

    self.time_dep_w = opt['time_dep_w'] #["struct_gaus", "struct_decay", "unstruct"]
    self.time_dep_omega = opt['time_dep_omega'] #["struct", "unstruct"]
    self.time_dep_q = opt['time_dep_q'] #["struct", "unstruct"]
    if self.time_dep_w in ["unstruct"]:
      self.num_timesteps = math.ceil(self.opt['time']/self.opt['step_size'])
    else:
      self.num_timesteps = 1

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
      if self.time_dep_omega in ["unstruct","struct"]:
        self.om_W = Parameter(torch.Tensor(self.num_timesteps, in_features))
        if self.time_dep_omega in ["struct"]:
          self.lamb_scales_Omega = Parameter(torch.Tensor(in_features, opt['num_lamb_omega']), requires_grad=True)
          self.lamb_starts_Omega = Parameter(torch.Tensor(in_features, opt['num_lamb_omega']), requires_grad=True)
          self.lamb_widths_Omega = Parameter(torch.Tensor(in_features, opt['num_lamb_omega']), requires_grad=True)
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

    #init source term params
    if self.opt['source_term'] == 'scalar':
      # self.q_scalar = nn.Parameter(torch.Tensor([1.]))
      self.q_scalar = nn.Parameter(torch.Tensor([0.]))
    elif self.opt['source_term'] == 'fidelity':
      self.q_fidelity = nn.Parameter(torch.Tensor([1.]))
    elif self.opt['source_term'] in ['diag', 'bias']:
      self.q_diag = nn.Parameter(torch.Tensor(in_features))
    elif self.opt['source_term'] in ['time_dep_bias']:
      self.q_diag_T = nn.Parameter(torch.Tensor(self.num_timesteps, in_features))
    elif self.opt['source_term'] == 'time_dep_q':
      if self.time_dep_q in ["struct"]:
        self.lamb_scales_Q = Parameter(torch.Tensor(in_features, opt['num_lamb_q']), requires_grad=True)
        self.lamb_starts_Q = Parameter(torch.Tensor(in_features, opt['num_lamb_q']), requires_grad=True)
        self.lamb_widths_Q = Parameter(torch.Tensor(in_features, opt['num_lamb_q']), requires_grad=True)
      elif self.time_dep_q in ["unstruct"]:
        nts = math.ceil(self.opt['time'] / self.opt['step_size'])
        self.gnl_Q_D_T = nn.Parameter(torch.Tensor(nts, in_features))

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

    if self.time_dep_w in ["unstruct", "struct_gaus", "struct_decay"]:
      self.init_W_timedep(in_features, opt)
    else:
      self.init_W(in_features, opt)

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

    if self.opt['conv_batch_norm'] == "shared":
      self.batchnorm_h = nn.BatchNorm1d(in_features)  # for zinc https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/gcn_layer.py
    elif self.opt['conv_batch_norm'] == "layerwise":
      nts = math.ceil(self.opt['time'] / self.opt['step_size'])
      self.batchnorms = [nn.BatchNorm1d(in_features).to(device) for _ in range(nts)]

    if self.opt['gnl_style'] in ['general_graph']:
      if self.opt['gnl_activation'] == "perona_malik":
        self.pm_a = Parameter(torch.Tensor([1.]))
        self.pm_b = Parameter(torch.Tensor([1.]))
      elif self.opt['gnl_activation'] == "pm_gaussian":
        self.pm_g = Parameter(torch.Tensor([1.]))
        self.pm_m = Parameter(torch.Tensor([1.]))
        self.pm_s = Parameter(torch.Tensor([1.]))
      elif self.opt['gnl_activation'] == "pm_invsq":
        self.pm_g = Parameter(torch.Tensor([1.]))
        self.pm_m = Parameter(torch.Tensor([1.]))
        self.pm_s = Parameter(torch.Tensor([1.]))
        # nts = math.ceil(self.opt['time'] / self.opt['step_size'])
        # self.pm_g = Parameter(torch.Tensor(nts))
        # self.pm_m = Parameter(torch.Tensor(nts))
        # self.pm_s = Parameter(torch.Tensor(nts))
      elif self.opt['gnl_activation'] == "pm_mlp":
        # self.pm_MLP = MLP(1, 32, 1, self.opt)
        self.layer1 = nn.Linear(1, 64).to(device)
        self.layer2 = nn.Linear(64, 1, bias=False).to(device)  # do not use bias on second layer
    elif self.opt['gnl_style'] in ['att_rep_laps']:
      if self.opt['gnl_activation'] == "perona_malik":
        self.pm_a = Parameter(torch.Tensor([1.]))
        self.pm_b = Parameter(torch.Tensor([1.]))
        self.anti_pm_a = Parameter(torch.Tensor([1.]))
        self.anti_pm_b = Parameter(torch.Tensor([1.]))
      elif self.opt['gnl_activation'] == "pm_gaussian":
        self.pm_g = Parameter(torch.Tensor([1.]))
        self.pm_m = Parameter(torch.Tensor([1.]))
        self.pm_s = Parameter(torch.Tensor([1.]))
        self.anti_pm_g = Parameter(torch.Tensor([1.]))
        self.anti_pm_m = Parameter(torch.Tensor([1.]))
        self.anti_pm_s = Parameter(torch.Tensor([1.]))
      elif self.opt['gnl_activation'] == "pm_invsq":
        self.pm_g = Parameter(torch.Tensor([1.]))
        self.pm_m = Parameter(torch.Tensor([1.]))
        self.pm_s = Parameter(torch.Tensor([1.]))
        self.anti_pm_g = Parameter(torch.Tensor([1.]))
        self.anti_pm_m = Parameter(torch.Tensor([1.]))
        self.anti_pm_s = Parameter(torch.Tensor([1.]))

    if self.time_dep_w in ["unstruct", "struct_gaus", "struct_decay"]:
      self.reset_W_timedep_parameters()
    else:
      self.reset_W_parameters()
    self.reset_Omega_parameters()
    self.reset_nonlinG_parameters()


  def init_W(self, in_features, opt):
    if self.opt['gnl_style'] in ['general_graph', 'attention_flavour']:
      # gnl_omega -> "gnl_W"
      if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod', 'free']:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))

      elif self.opt['gnl_W_style'] == 'tri':
        # init an upper triangular and a diagonal vector
        self.W_W = Parameter(torch.Tensor(int((in_features - 1) * in_features / 2)))
        self.W_D = Parameter(torch.Tensor(in_features))

      elif self.opt['gnl_W_style'] == 'diag':
        if self.opt['gnl_W_diag_init'] == 'linear':
          d_range = torch.tensor(list(range(in_features)), device=self.device)
          self.gnl_W_D = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (in_features - 1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
        else:
          self.gnl_W_D = Parameter(torch.ones(in_features), requires_grad=opt['gnl_W_param_free'])
      elif self.opt['gnl_W_style'] == 'diag_dom':
        self.W_W = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['gnl_W_param_free'])
        self.t_a = Parameter(torch.Tensor(in_features), requires_grad=opt['gnl_W_param_free'])
        self.r_a = Parameter(torch.Tensor(in_features), requires_grad=opt['gnl_W_param_free'])

      elif self.opt['gnl_W_style'] == 'k_diag_pc':
        k_num = int(self.opt['k_diag_pc'] * in_features)
        if k_num % 2 == 0:
          k_num += 1
        k_num = min(k_num, in_features)
        self.gnl_W_diags = Parameter(torch.Tensor(in_features, k_num))
      elif self.opt['gnl_W_style'] == 'k_block':
        assert opt['k_blocks'] * opt['block_size'] <= in_features, 'blocks exceeded hidden dim'
        self.gnl_W_blocks = Parameter(torch.Tensor(opt['k_blocks'] * opt['block_size'], opt['block_size']))
        self.gnl_W_D = Parameter(torch.Tensor(in_features - opt['k_blocks'] * opt['block_size']))
      elif self.opt['gnl_W_style'] == 'k_diag':
        assert opt['k_diags'] % 2 == 1 and opt['k_diags'] <= in_features, 'must have odd number of k diags'
        self.gnl_W_diags = Parameter(
          torch.Tensor(in_features, opt['k_diags']))  # or (2k-1) * n + k * (k - 1) if don't wrap around
      elif self.opt['gnl_W_style'] in ['Z_diag']:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))
      elif self.opt['gnl_W_style'] in ['GS', 'GS_Z_diag']:
        self.W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] in ['cgnn', 'cgnn_Z_diag']:
        self.W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'loss_W_orthog':
        self.gnl_W_U = Parameter(torch.Tensor(in_features, in_features))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'W_orthog_init':
        W_U = torch.rand((in_features, in_features), device=self.device)
        W_GS = gram_schmidt(W_U)
        self.gnl_W_U = Parameter(W_GS)
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'householder':
        # init L=d-1 x d-dim random vectors
        self.gnl_W_hh = Parameter(torch.Tensor(in_features, self.opt['householder_L']))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'skew_sym':
        # init an upper triangular and a diagonal vector
        self.gnl_W_ss = Parameter(torch.ones(int((in_features ** 2 - in_features) / 2)))
        self.gnl_W_D = Parameter(torch.ones(in_features))
      elif self.opt['gnl_W_style'] == 'feature':
        self.Om_phi = Parameter(torch.Tensor(in_features))
        self.W_psi = Parameter(torch.Tensor(in_features))
      elif self.opt['gnl_W_style'] == 'positional':
        self.phi = nn.Linear(self.opt['pos_enc_hidden_dim'], self.in_features)
        self.psi = nn.Linear(self.opt['pos_enc_hidden_dim'], self.in_features)

      #inits for simple 2hops
      if self.opt['two_hops']:
        if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod', 'free']:
          self.W_W2 = Parameter(torch.Tensor(in_features, in_features))
        elif self.opt['gnl_W_style'] == 'tri':
          # init an upper triangular and a diagonal vector
          self.W_W2 = Parameter(torch.Tensor(int((in_features - 1) * in_features / 2)))
          self.W_D2 = Parameter(torch.Tensor(in_features))
        elif self.opt['gnl_W_style'] == 'diag':
          if self.opt['gnl_W_diag_init'] == 'linear':
            d_range = torch.tensor(list(range(in_features)), device=self.device)
            self.gnl_W_D2 = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (in_features - 1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
          else:
            self.gnl_W_D2 = Parameter(torch.ones(in_features), requires_grad=opt['gnl_W_param_free'])
        elif self.opt['gnl_W_style'] == 'diag_dom':
          self.W_W2 = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['gnl_W_param_free'])
          self.t_a2 = Parameter(torch.Tensor(in_features), requires_grad=opt['gnl_W_param_free'])
          self.r_a2 = Parameter(torch.Tensor(in_features), requires_grad=opt['gnl_W_param_free'])

    elif self.opt['gnl_style'] == 'att_rep_laps':
      if self.opt['gnl_W_style'] == 'att_rep_lap_block':
        half_in_features = int(in_features / 2)
        self.L_W = Parameter(torch.Tensor(half_in_features, half_in_features - 1),
                             requires_grad=opt['gnl_W_param_free'])
        self.L_t_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
        self.L_r_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
        self.R_W = Parameter(torch.Tensor(half_in_features, half_in_features - 1),
                             requires_grad=opt['gnl_W_param_free'])
        self.R_t_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
        self.R_r_a = Parameter(torch.Tensor(half_in_features), requires_grad=opt['gnl_W_param_free'])
      elif self.opt['gnl_W_style'] == 'sum':
        self.W_W = Parameter(torch.Tensor(in_features, in_features))

  def init_W_timedep(self, in_features, opt):
    if self.opt['gnl_style'] in ['general_graph', 'attention_flavour']:
      # gnl_omega -> "gnl_W"
      if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod', 'free']:
        if self.time_dep_w in ["unstruct"]:
          self.W_W_T = Parameter(torch.Tensor(self.num_timesteps, in_features, in_features))

      elif self.opt['gnl_W_style'] == 'tri':
        if self.time_dep_w in ["unstruct"]:
          # init an upper triangular and a diagonal vector
          self.W_W_T = Parameter(torch.Tensor(self.num_timesteps, int((in_features - 1) * in_features / 2)))
          self.W_D_T = Parameter(torch.Tensor(self.num_timesteps, in_features))

      elif self.opt['gnl_W_style'] == 'diag':
        if self.time_dep_w in ["unstruct"]:
          if self.opt['gnl_W_diag_init'] == 'linear':
            d_range = torch.tensor([list(range(in_features)) for _ in range(self.num_timesteps)], device=self.device)
            self.gnl_W_D_T = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (in_features - 1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
          else:
            self.gnl_W_D_T = Parameter(torch.ones(self.num_timesteps, in_features), requires_grad=opt['gnl_W_param_free'])

        elif self.time_dep_w in ["struct"]:
          self.brt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
          self.crt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
          self.drt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)

          if self.opt['gnl_W_diag_init'] == 'linear':
            d_range = torch.tensor(list(range(in_features)), device=self.device)
            self.gnl_W_D = Parameter(self.opt['gnl_W_diag_init_q'] * d_range / (in_features - 1) + self.opt['gnl_W_diag_init_r'], requires_grad=opt['gnl_W_param_free'])
          else:
            self.gnl_W_D = Parameter(torch.ones(in_features), requires_grad=opt['gnl_W_param_free'])


        elif self.opt['gnl_W_style'] == 'diag_dom':
          self.W_W = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['gnl_W_param_free'])
          if self.time_dep_w in ["unstruct"]:
            self.t_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
            self.r_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
          elif self.time_dep_w in ["struct"]:
            self.at = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
            self.bt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
            self.gt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)

        elif self.opt['gnl_W_style'] in ['Z_diag', 'GS_Z_diag', 'cgnn_Z_diag', 'loss_W_orthog', 'W_orthog_init', 'householder', 'skew_sym']:
          if self.time_dep_w in ["struct_gaus"]:
            self.lamb_scales_W = Parameter(torch.Tensor(in_features, opt['num_lamb_w']), requires_grad=True)
            self.lamb_starts_W = Parameter(torch.Tensor(in_features, opt['num_lamb_w']), requires_grad=True)
            self.lamb_widths_W = Parameter(torch.Tensor(in_features, opt['num_lamb_w']), requires_grad=True)
          elif self.time_dep_w in ["struct_decay"]:
            self.lambs_W = Parameter(torch.Tensor(in_features), requires_grad=True)
            self.lamb_decays_W = Parameter(torch.Tensor(in_features), requires_grad=True)

          nts = self.num_timesteps if self.time_dep_w == "unstruct" else 1
          if self.opt['gnl_W_style'] in ['Z_diag']:
            self.W_W_T = Parameter(torch.Tensor(nts, in_features, in_features))
          else:
            self.gnl_W_D_T = Parameter(torch.ones(nts, in_features))
            if self.opt['gnl_W_style'] in ['GS_Z_diag', 'cgnn_Z_diag']:
              self.W_U = Parameter(torch.Tensor(in_features, in_features))
            elif self.opt['gnl_W_style'] in ['loss_W_orthog']:
              self.gnl_W_U = Parameter(torch.Tensor(in_features, in_features))
            elif self.opt['gnl_W_style'] == 'W_orthog_init':
              W_U = torch.rand((in_features, in_features), device=self.device)
              W_GS = gram_schmidt(W_U)
              self.gnl_W_U = Parameter(W_GS)
            elif self.opt['gnl_W_style'] == 'householder':
              # init L=d-1 x d-dim random vectors
              self.gnl_W_hh = Parameter(torch.Tensor(in_features, self.opt['householder_L']))
            elif self.opt['gnl_W_style'] == 'skew_sym':
              # init an upper triangular and a diagonal vector
              self.gnl_W_ss = Parameter(torch.ones(int((in_features ** 2 - in_features) / 2)))

  def reset_Omega_parameters(self):
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
      if self.time_dep_omega in ["struct"]:
        uniform(self.lamb_scales_Omega, a=-0.1, b=0.1) #Omega can be smaller than W and Q
        # uniform(self.lamb_starts_Omega, a=-1, b=1)
        uniform(self.lamb_starts_Omega, a=0, b=np.sqrt(self.opt['time']))
        uniform(self.lamb_widths_Omega, a=-1, b=1)
    elif self.opt['gnl_omega'] == 'Omega_W_eig':
      uniform(self.om_W, a=-1, b=1)

  def reset_nonlinG_parameters(self):
    if self.opt['gnl_measure'] in ['deg_poly', 'deg_poly_exp']:
      ones(self.m_alpha)
      ones(self.m_beta)
      ones(self.m_gamma)
    elif self.opt['gnl_measure'] in ['nodewise']:
      ones(self.measure)
    elif self.opt['gnl_measure'] in ['nodewise_exp']:
      zeros(self.measure)

    if self.opt['source_term'] == 'scalar':
      # ones(self.q_scalar)
      zeros(self.q_scalar)
    elif self.opt['source_term'] == 'fidelity':
      ones(self.q_fidelity)
    elif self.opt['source_term'] == 'diag':
      ones(self.q_diag)
    elif self.opt['source_term'] == 'bias':
      uniform(self.q_diag, a=-0.1, b=0.1)
    elif self.opt['source_term'] in ['time_dep_bias']:
      # uniform(self.q_diag_T, a=-0.1, b=0.1)
      zeros(self.q_diag_T)
    elif self.opt['source_term'] == 'time_dep_q':
      if self.time_dep_q in ["struct"]:
        uniform(self.lamb_scales_Q, a=-1, b=1)
        # uniform(self.lamb_starts_Q, a=-1, b=1)
        uniform(self.lamb_starts_Q, a=0, b=np.sqrt(self.opt['time']))
        uniform(self.lamb_widths_Q, a=-1, b=1)
      elif self.time_dep_q in ["unstruct"]:
        xavier_uniform_(self.gnl_Q_D_T)

    if self.opt['gnl_style'] in ['general_graph']:
      if self.opt['gnl_activation'] == "perona_malik":
        ones(self.pm_a)
        ones(self.pm_b)
      elif self.opt['gnl_activation'] == "pm_gaussian":
        ones(self.pm_g)
        ones(self.pm_m)
        ones(self.pm_s)
      elif self.opt['gnl_activation'] == "pm_invsq":
        ones(self.pm_g)
        ones(self.pm_m)
        ones(self.pm_s)

  def reset_W_parameters(self):
    # W's
    if self.opt['gnl_style'] in ['general_graph', 'attention_flavour']:
      if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod', 'free', 'Z_diag']:
        # glorot(self.W_W)
        xavier_uniform_(self.W_W)
      elif self.opt['gnl_W_style'] == 'tri':
        uniform(self.W_W, a=-1, b=1)
        uniform(self.W_D, a=-1, b=1)

      elif self.opt['gnl_W_style'] == 'diag':
        if self.opt['gnl_W_diag_init'] == 'uniform':
          uniform(self.gnl_W_D, a=-1, b=1)
        elif self.opt['gnl_W_diag_init'] == 'identity':
          ones(self.gnl_W_D)
        elif self.opt['gnl_W_diag_init'] == 'linear':
          pass  # done in init
      elif self.opt['gnl_W_style'] == 'diag_dom':
        if self.opt['gnl_W_diag_init'] == 'uniform':
          glorot(self.W_W)
          uniform(self.t_a, a=-1, b=1)
          uniform(self.r_a, a=-1, b=1)
        elif self.opt['gnl_W_diag_init'] == 'identity':
          zeros(self.W_W)
          constant(self.t_a, fill_value=1)
          constant(self.r_a, fill_value=1)
        elif self.opt['gnl_W_diag_init'] == 'linear':
          glorot(self.W_W)
          constant(self.t_a, fill_value=self.opt['gnl_W_diag_init_q'])
          constant(self.r_a, fill_value=self.opt['gnl_W_diag_init_r'])
      elif self.opt['gnl_W_style'] == 'k_block':
        glorot(self.gnl_W_blocks)
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'k_diag':
        uniform(self.gnl_W_diags, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'k_diag_pc':
        if self.opt['gnl_W_diag_init'] == 'uniform':
          uniform(self.gnl_W_diags, a=-1, b=1)
        elif self.opt['gnl_W_diag_init'] == 'identity':
          ones(self.gnl_W_diags)
      elif self.opt['gnl_W_style'] in ['loss_W_orthog']:
        glorot(self.gnl_W_U)
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] in ['GS', 'GS_Z_diag']:
        glorot(self.W_U)
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] in ['cgnn', 'cgnn_Z_diag']:
        glorot(self.W_U)
      elif self.opt['gnl_W_style'] == 'W_orthog_init':
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'householder':
        xavier_uniform_(self.gnl_W_hh)
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'skew_sym':
        uniform(self.gnl_W_ss, a=-1, b=1)
        uniform(self.gnl_W_D, a=-1, b=1)
      elif self.opt['gnl_W_style'] == 'feature':
        glorot(self.Om_phi)
        glorot(self.W_psi)
      elif self.opt['gnl_W_style'] == 'positional':
        pass  # linear layer

      #inits for simple 2hops
      if self.opt['two_hops']:
        if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod', 'free', 'Z_diag']:
          xavier_uniform_(self.W_W2)
        elif self.opt['gnl_W_style'] == 'tri':
          uniform(self.W_W2, a=-1, b=1)
          uniform(self.W_D2, a=-1, b=1)
        elif self.opt['gnl_W_style'] == 'diag':
          if self.opt['gnl_W_diag_init'] == 'uniform':
            uniform(self.gnl_W_D2, a=-1, b=1)
          elif self.opt['gnl_W_diag_init'] == 'identity':
            ones(self.gnl_W_D2)
          elif self.opt['gnl_W_diag_init'] == 'linear':
            pass  # done in init
        elif self.opt['gnl_W_style'] == 'diag_dom':
          if self.opt['gnl_W_diag_init'] == 'uniform':
            glorot(self.W_W2)
            uniform(self.t_a2, a=-1, b=1)
            uniform(self.r_a2, a=-1, b=1)
          elif self.opt['gnl_W_diag_init'] == 'identity':
            zeros(self.W_W2)
            constant(self.t_a2, fill_value=1)
            constant(self.r_a2, fill_value=1)
          elif self.opt['gnl_W_diag_init'] == 'linear':
            glorot(self.W_W2)
            constant(self.t_a2, fill_value=self.opt['gnl_W_diag_init_q'])
            constant(self.r_a2, fill_value=self.opt['gnl_W_diag_init_r'])

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

  def reset_W_timedep_parameters(self): #todo question do all these init's work for 3d tensors?
    # W's
    if self.opt['gnl_style'] in ['general_graph', 'attention_flavour']:
      if self.opt['gnl_W_style'] in ['sum', 'prod', 'neg_prod', 'free']:
        # glorot(self.W_W_T)
        xavier_uniform_(self.W_W_T)

      elif self.opt['gnl_W_style'] == 'tri':
        uniform(self.W_W_T, a=-1, b=1)
        uniform(self.W_D_T, a=-1, b=1)

      elif self.opt['gnl_W_style'] == 'diag':
        if self.time_dep_w == "unstruct":
          if self.opt['gnl_W_diag_init'] == 'uniform':
            uniform(self.gnl_W_D_T, a=-1, b=1)
          elif self.opt['gnl_W_diag_init'] == 'identity':
            ones(self.gnl_W_D_T)
        elif self.time_dep_w == "struct":
          uniform(self.brt, a=-1, b=1)
          uniform(self.crt, a=-1, b=1)
          uniform(self.drt, a=-1, b=1)
          if self.opt['gnl_W_diag_init'] == 'uniform':
            uniform(self.gnl_W_D, a=-1, b=1)
          elif self.opt['gnl_W_diag_init'] == 'identity':
            ones(self.gnl_W_D)
          elif self.opt['gnl_W_diag_init'] == 'linear':
            pass  # done in init

      elif self.opt['gnl_W_style'] == 'diag_dom':
        if self.time_dep_w == "struct":
          uniform(self.at, a=-1, b=1)
          uniform(self.bt, a=-1, b=1)
          uniform(self.gt, a=-1, b=1)
        if self.time_dep_w == "struct":
          if self.opt['gnl_W_diag_init'] == 'uniform':
            glorot(self.W_W)
            uniform(self.t_a, a=-1, b=1)
            uniform(self.r_a, a=-1, b=1)
          elif self.opt['gnl_W_diag_init'] == 'identity':
            zeros(self.W_W)
            constant(self.t_a, fill_value=1)
            constant(self.r_a, fill_value=1)
          elif self.opt['gnl_W_diag_init'] == 'linear':
            glorot(self.W_W)
            constant(self.t_a, fill_value=self.opt['gnl_W_diag_init_q'])
            constant(self.r_a, fill_value=self.opt['gnl_W_diag_init_r'])

      elif self.opt['gnl_W_style'] in ['Z_diag', 'GS_Z_diag', 'cgnn_Z_diag', 'loss_W_orthog', 'W_orthog_init', 'householder', 'skew_sym']:
        if self.time_dep_w == "struct_gaus":
          uniform(self.lamb_scales_W, a=-1, b=1)
          # uniform(self.lamb_starts_W, a=-1, b=1)
          uniform(self.lamb_starts_W, a=0, b=np.sqrt(self.opt['time']))
          uniform(self.lamb_widths_W, a=-1, b=1)
        elif self.time_dep_w in ["struct_decay"]:
          uniform(self.lambs_W, a=-1, b=1)
          uniform(self.lamb_decays_W, a=-1, b=1)

        if self.opt['gnl_W_style'] in ['Z_diag']:
          glorot(self.W_W_T)
        else:
          uniform(self.gnl_W_D_T, a=-1, b=1)
          if self.opt['gnl_W_style'] in ['loss_W_orthog']:
            glorot(self.gnl_W_U)
          elif self.opt['gnl_W_style'] in ['GS_Z_diag', 'cgnn_Z_diag']:
            glorot(self.W_U)
          elif self.opt['gnl_W_style'] == 'W_orthog_init':
            pass
          elif self.opt['gnl_W_style'] == 'householder':
            xavier_uniform_(self.gnl_W_hh)
          elif self.opt['gnl_W_style'] == 'skew_sym':
            uniform(self.gnl_W_ss, a=-1, b=1) #is vector can't xavier


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
      if self.time_dep_omega in ["unstruct", "struct"]:
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

  def set_gnlWS(self):
    "note every W is made symetric before returning"
    if self.opt['gnl_W_style'] in ['prod']:
      return self.W_W @ self.W_W.t()
    elif self.opt['gnl_W_style'] in ['neg_prod']:
      return -self.W_W @ self.W_W.t()
    elif self.opt['gnl_W_style'] in ['sum']:
      return (self.W_W + self.W_W.t()) / 2
    elif self.opt['gnl_W_style'] in ['free']:
      return self.W_W
    elif self.opt['gnl_W_style'] in ['tri']:
      #init an upper triangular and a diagonal vector
      tri = []
      start = 0
      for i in range(self.in_features):
        tri.append(torch.cat((torch.zeros(1+i, device=self.device), self.W_W[start: start + self.in_features - (1+i)])))
        start += self.in_features - (1+i)
      tri = torch.stack(tri, dim=0)
      W_hat = (tri + tri.T) / 2 + torch.diag(self.W_D)
      return W_hat

    elif self.opt['gnl_W_style'] in ['Z_diag']:
      return (self.W_W + self.W_W.t()) / 2
    elif self.opt['gnl_W_style'] == 'diag':
      return torch.diag(self.gnl_W_D)
    elif self.opt['gnl_W_style'] == 'diag_dom':
      W_temp = torch.cat([self.W_W, torch.zeros((self.in_features, 1), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      W = (W+W.T) / 2
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
      self.gnl_W_U = gram_schmidt(self.W_U)
      # W_D = torch.tanh(self.gnl_W_D) #
      # W_D = torch.clamp(self.gnl_W_D, min=-1, max=1)
      W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
      return W_hat
    elif self.opt['gnl_W_style'] in ['cgnn', 'cgnn_Z_diag']: # https://github.com/DeepGraphLearning/ContinuousGNN/blob/a82332eb5d85e80fd233ab35e4d155f34eb1275d/src/trainer.py#L108
      beta = self.opt['W_beta']
      with torch.no_grad():         # https://stackoverflow.com/questions/62198351/why-doesnt-pytorch-allow-inplace-operations-on-leaf-variables
        W_U = self.W_U.clone()
        W_U = self.W_U.copy_((1 + beta) * W_U - beta * W_U @ W_U.t() @ W_U)
      self.gnl_W_U = W_U
      #choose not to restrict spectrum
      # W_D = torch.clamp(self.gnl_W_D, min=-1, max=1) #self.gnl_W_D
      # self.gnl_W_D = torch.tanh(self.gnl_W_D) #self.gnl_W_D
      W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
      return W_hat
    elif self.opt['gnl_W_style'] == 'loss_W_orthog':
      W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
      return W_hat
    elif self.opt['gnl_W_style'] == 'W_orthog_init':
      W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
      return W_hat
    elif self.opt['gnl_W_style'] == 'householder':
      #init L=d-1 x d-dim random vectors #https://github.com/riannevdberg/sylvester-flows/blob/32dde9b7d696fee94f946a338182e542779eecfe/models/VAE.py#L502
      H_prod = torch.eye(self.in_features, device=self.device)
      for l in range(self.opt['householder_L']):
        v = self.gnl_W_hh[:,l]
        H = torch.eye(self.in_features, device=self.device) - 2 * torch.outer(v,v) / torch.pow(torch.norm(v, p=2), 2)
        H_prod = H_prod @ H
      self.gnl_W_U = H_prod
      W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
      return W_hat
    elif self.opt['gnl_W_style'] == 'skew_sym':
      #init an upper triangular and a diagonal vector
      SS = []
      start = 0
      for i in range(self.in_features):
        SS.append(torch.cat((torch.zeros(1+i, device=self.device), self.gnl_W_ss[start: start + self.in_features - (1+i)])))
        start += self.in_features - (1+i)
      S = torch.stack(SS, dim=0)
      S = (S - S.T) / 2
      self.gnl_W_U = torch.linalg.inv(torch.eye(self.in_features, device=self.device) - 0.5 * self.opt['step_size'] * S) @ (torch.eye(self.in_features, device=self.device) + 0.5 * self.opt['step_size'] * S)
      W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
      return W_hat

    elif self.opt['gnl_W_style'] == 'identity':
      return torch.eye(self.in_features, device=self.device)
    elif self.opt['gnl_W_style'] == 'feature':
      pass
    elif self.opt['gnl_W_style'] == 'positional':
      pass

    elif self.opt['gnl_W_style'] == 'att_rep_lap_block':
      half_in_features = int(self.in_features / 2)
      L_temp = torch.cat([self.L_W, torch.zeros((half_in_features, 1), device=self.device)], dim=1)
      L = torch.stack([torch.roll(L_temp[i], shifts=i+1, dims=-1) for i in range(half_in_features)])
      L = (L+L.T) / 2
      L_sum = (self.L_t_a**2 + 1) * torch.abs(L).sum(dim=1) + self.L_r_a**2
      L_block = L + torch.diag(L_sum)
      Ws = torch.zeros((self.in_features, self.in_features), device=self.device)
      Ws[0:half_in_features, 0:half_in_features] = L_block

      R_temp = torch.cat([self.R_W, torch.zeros((half_in_features, 1), device=self.device)], dim=1)
      R = torch.stack([torch.roll(R_temp[i], shifts=i+1, dims=-1) for i in range(half_in_features)])
      R = (R+R.T) / 2

      R_sum = (self.R_t_a**2 + 1) * torch.abs(R).sum(dim=1) + self.R_r_a**2
      R_block = R + torch.diag(R_sum)
      R_Ws = torch.zeros((self.in_features, self.in_features), device=self.device)
      R_Ws[half_in_features:, half_in_features:] = R_block

      return Ws, R_Ws

  def set_gnlWS2(self):
    "note every W is made symetric before returning"
    if self.opt['gnl_W_style'] in ['prod']:
      return self.W_W2 @ self.W_W2.t()
    elif self.opt['gnl_W_style'] in ['neg_prod']:
      return -self.W_W2 @ self.W_W2.t()
    elif self.opt['gnl_W_style'] in ['sum']:
      return (self.W_W2 + self.W_W2.t()) / 2
    elif self.opt['gnl_W_style'] in ['free']:
      return self.W_W2
    elif self.opt['gnl_W_style'] in ['tri']:
      #init an upper triangular and a diagonal vector
      tri = []
      start = 0
      for i in range(self.in_features):
        tri.append(torch.cat((torch.zeros(1+i, device=self.device), self.W_W2[start: start + self.in_features - (1+i)])))
        start += self.in_features - (1+i)
      tri = torch.stack(tri, dim=0)
      W_hat = (tri + tri.T) / 2 + torch.diag(self.W_D2)
      return W_hat
    elif self.opt['gnl_W_style'] in ['Z_diag']:
      return (self.W_W2 + self.W_W2.t()) / 2
    elif self.opt['gnl_W_style'] == 'diag':
      return torch.diag(self.gnl_W_D2)
    elif self.opt['gnl_W_style'] == 'diag_dom':
      W_temp = torch.cat([self.W_W2, torch.zeros((self.in_features, 1), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      W = (W+W.T) / 2
      W_sum = self.t_a2 * torch.abs(W).sum(dim=1) + self.r_a2
      Ws = W + torch.diag(W_sum)
      return Ws

  def set_gnlOmega(self):
    if self.opt['gnl_omega'] == 'diag':
      if self.opt['gnl_omega_diag'] == 'free':
        Omega = torch.diag(self.om_W)
        # todo check if this activation is still needed
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

  def set_gnlWS_timedep(self, T):
    "note every W is made symetric before returning"
    if self.opt['gnl_W_style'] in ['sum']:
      return (self.W_W_T[T] + self.W_W_T[T].t()) / 2
    elif self.opt['gnl_W_style'] in ['prod']:
      return self.W_W_T[T] @ self.W_W_T[T].t()
    elif self.opt['gnl_W_style'] in ['neg_prod']:
      return -self.W_W_T[T] @ self.W_W_T[T].t()
    elif self.opt['gnl_W_style'] in ['free']:
      return self.W_W_T[T]

    elif self.opt['gnl_W_style'] in ['tri']:
      #init an upper triangular and a diagonal vector
      tri = []
      start = 0
      for i in range(self.in_features):
        tri.append(torch.cat((torch.zeros(1+i, device=self.device), self.W_W_T[T][start: start + self.in_features - (1+i)])))
        start += self.in_features - (1+i)
      tri = torch.stack(tri, dim=0)
      W_hat = (tri + tri.T) / 2 + torch.diag(self.W_D_T[T])
      return W_hat

    elif self.opt['gnl_W_style'] == 'diag':
      if self.time_dep_w == "unstruct":
        return torch.diag(self.gnl_W_D_T[T])
      elif self.time_dep_w == "struct":
        W = self.gnl_W_D
        alpha = torch.diag(torch.exp(self.brt[T] * T + self.brt[T]))
        beta = torch.diag(torch.exp(-self.brt[T] * T - self.crt[T]) + self.drt[T])
        Wplus = torch.diag(F.relu(W))
        Wneg = torch.diag(-1. * F.relu(-W))
        return alpha @ Wplus - beta @ Wneg

    elif self.opt['gnl_W_style'] == 'diag_dom':
      W_temp = torch.cat([self.W_W, torch.zeros((self.in_features, 1), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      W = (W+W.T) / 2
      if self.time_dep_w == "unstruct":
        W_sum = self.t_a[T] * torch.abs(W).sum(dim=1) + self.r_a[T]
      elif self.time_dep_w == "struct":
        W_sum = W + self.at[T] * F.tanh(self.bt[T] * T + self.gt[T]) * torch.eye(n=W.shape[0], m=W.shape[1], device=self.device)
      Ws = W + torch.diag(W_sum)
      return Ws

    # idea for these options time dependent just always add the leading time dimension but if it's structured like abovecan just set T=0
    # then just set in the sertW_t function
    elif self.opt['gnl_W_style'] in ['Z_diag']: #todo check if ever set self.gnl_W_D(T) = eval(W(T))
      return (self.W_W_T[T, ...] + self.W_W_T[T, ...].t()) / 2

    elif self.opt['gnl_W_style'] in ['GS_Z_diag', 'cgnn_Z_diag', 'loss_W_orthog', 'W_orthog_init', 'householder', 'skew_sym']:
      #update diagonals with eigen values
      if self.time_dep_w == "unstruct":
        self.gnl_W_D = self.gnl_W_D_T[T]
      elif self.time_dep_w == "struct_gaus":
        self.gnl_W_D = self.gaussian_lin_comb(T, self.lamb_scales_W, self.lamb_starts_W, self.lamb_widths_W)
      elif self.time_dep_w == "struct_decay":
        self.gnl_W_D = self.lambs_W * torch.exp(-self.lamb_decays_W**2 * T)

      #calc orthogonals (for T-=0) and return W
      if self.opt['gnl_W_style'] in ['GS_Z_diag']:
        if T == 0:
          self.gnl_W_U = gram_schmidt(self.W_U)
        W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
        return W_hat
      elif self.opt['gnl_W_style'] in ['cgnn_Z_diag']:  # https://github.com/DeepGraphLearning/ContinuousGNN/blob/a82332eb5d85e80fd233ab35e4d155f34eb1275d/src/trainer.py#L108
        pass #can't do cgnn time dep because changing leafs in the integration
        # beta = self.opt['W_beta']
        # with torch.no_grad(): # https://stackoverflow.com/questions/62198351/why-doesnt-pytorch-allow-inplace-operations-on-leaf-variables
        #   W_U = self.W_U.clone()
        #   W_U = self.W_U.copy_((1 + beta) * W_U - beta * W_U @ W_U.t() @ W_U)
        # self.gnl_W_U = W_U
        # W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
        # return W_hat
      elif self.opt['gnl_W_style'] == 'loss_W_orthog':
        W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
        return W_hat
      elif self.opt['gnl_W_style'] == 'W_orthog_init':
        W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
        return W_hat
      elif self.opt['gnl_W_style'] == 'householder':
        #init L=d-1 x d-dim random vectors #https://github.com/riannevdberg/sylvester-flows/blob/32dde9b7d696fee94f946a338182e542779eecfe/models/VAE.py#L502
        if T == 0:
          H_prod = torch.eye(self.in_features, device=self.device)
          for l in range(self.opt['householder_L']):
            v = self.gnl_W_hh[:,l]
            H = torch.eye(self.in_features, device=self.device) - 2 * torch.outer(v,v) / torch.pow(torch.norm(v, p=2), 2)
            H_prod = H_prod @ H
          self.gnl_W_U = H_prod
        W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
        return W_hat
      elif self.opt['gnl_W_style'] == 'skew_sym':
        #init an upper triangular and a diagonal vector
        if T == 0:
          SS = []
          start = 0
          for i in range(self.in_features):
            SS.append(torch.cat((torch.zeros(1+i, device=self.device), self.gnl_W_ss[start: start + self.in_features - (1+i)])))
            start += self.in_features - (1+i)
          S = torch.stack(SS, dim=0)
          S = (S - S.T) / 2
          self.gnl_W_U = torch.linalg.inv(torch.eye(self.in_features, device=self.device) - 0.5 * self.opt['step_size'] * S) @ (torch.eye(self.in_features, device=self.device) + 0.5 * self.opt['step_size'] * S)
        W_hat = self.gnl_W_U @ torch.diag(self.gnl_W_D) @ self.gnl_W_U.t()
        return W_hat


  def set_gnlOmega_timedep(self, T):
    if self.opt['gnl_omega'] == 'diag':
      if self.time_dep_omega == "unstruct":
        Omega = torch.diag(self.om_W[T])
      elif self.time_dep_omega == "struct":
        Omega = self.gaussian_lin_comb(T, self.lamb_scales_Omega, self.lamb_starts_Omega, self.lamb_widths_Omega)
      #todo check if this activation is still needed given learned sign of gaussians
      if self.opt['gnl_omega_activation'] == 'exponential':
        Omega = -torch.exp(Omega)
      Omega = torch.diag(Omega)
    else:
      Omega = self.set_gnlOmega()
    return Omega

  def set_gnlQ_timedep(self, T):
    if self.time_dep_q == "unstruct":
      return self.gnl_Q_D_T[T]
    elif self.time_dep_q == "struct":
      return self.gaussian_lin_comb(T, self.lamb_scales_Q, self.lamb_starts_Q, self.lamb_widths_Q)

  def gaussian_lin_comb(self, T, lamb_scales, lamb_starts, lamb_widths):
    lambs_D = lamb_scales * torch.exp(-(T - lamb_starts**2) ** 2 * lamb_widths ** 2)
    return lambs_D.sum(-1)

  def skew_gaussian_lin_comb(self):
    pass
    #https://math.stackexchange.com/questions/1128781/whats-the-formula-for-the-probability-density-function-of-skewed-normal-distribu


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
      measure = torch.tensor([1], device=self.device) #1. #torch.ones(self.n_nodes, device=self.device) #torch.tensor([1.]) #torch.ones(x.shape[0], device=self.device)
      src_meas = torch.tensor([1], device=self.device) #1
      dst_meas = torch.tensor([1], device=self.device) #1
      measures_src_dst = torch.tensor([1], device=self.device) #1
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
    return f

  def predict(self, z):
    z = self.GNN_postXN(z)
    logits = self.GNN_m2(z)
    pred = logits.max(1)[1]
    return logits, pred

  def predict_graph(self, z):
    z = self.GNN_postXN(z)
    z = self.GNN_m2(z)
    if self.opt['graph_pool'] == "add":
      pred = global_add_pool(z, self.odeblock.odefunc.data.batch).squeeze(-1)
    elif self.opt['graph_pool'] == "mean":
      pred = global_mean_pool(z, self.odeblock.odefunc.data.batch).squeeze(-1)
    return pred

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

  def calc_dot_prod_attention(self, src_x, dst_x, x=None, T=0):
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
        attention = torch.tanh(fOmf)
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

    elif self.opt['gnl_style'] in ['general_graph']:
      # del src_x #trying to streamline for ogbn-arxiv
      # del dst_x
      # torch.cuda.empty_cache()
      # get degrees
      src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)

      # calc bilinear form
      if self.opt['gnl_activation'] != 'identity':
        fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                            dst_x * dst_deginvsqrt.unsqueeze(dim=1))
        if self.opt['gnl_activation'] == 'sigmoid':
          attention = torch.sigmoid(fOmf)
        elif self.opt['gnl_activation'] == "squareplus":
          attention = (fOmf + torch.sqrt(fOmf ** 2 + 4)) / 2
        elif self.opt['gnl_activation'] == "sigmoid_deriv":
          attention = sigmoid_deriv(fOmf)
        elif self.opt['gnl_activation'] == "tanh":
          attention = torch.tanh(fOmf)
        elif self.opt['gnl_activation'] == "tanh_deriv":
          attention = tanh_deriv(fOmf)
        elif self.opt['gnl_activation'] == "squareplus_deriv":
          attention = squareplus_deriv(fOmf)
        elif self.opt['gnl_activation'] == "exponential":
          attention = torch.exp(fOmf)
        elif self.opt['gnl_activation'] == "perona_malik":
          attention = self.pm_a**2 / (self.pm_a**2 + torch.exp(-self.pm_b**2 * fOmf))
        elif self.opt['gnl_activation'] == "pm_gaussian":
          attention = (self.pm_g**2 + 1) * torch.exp(-(fOmf-self.pm_m**2)**2/self.pm_s**2)
        elif self.opt['gnl_activation'] == "pm_invsq":
          attention = self.pm_g**2 / (1 + self.pm_m**2 * (fOmf - self.pm_s**2)**2)
          # attention = self.pm_g[T]**2 / (1 + self.pm_m[T]**2 * (fOmf - self.pm_s[T]**2)**2)
          #normalisation
          # attention = self.pm_g**2 / (1 + self.pm_m**2 * ((fOmf/(torch.abs(fOmf) + 1) - self.pm_s**2)**2))
          #delay and normalisation
          # if T > 4:
          #   mean_fOmf = fOmf.mean()
          #   attention = self.pm_g**2 / (1 + self.pm_m**2 * (fOmf - self.pm_s**2*mean_fOmf)**2)
          # else:
          #   attention = 1.
        elif self.opt['gnl_activation'] == "pm_mlp":
          # attention = self.pm_MLP(fOmf.unsqueeze(-1))
          # self.layer2(torch.relu(self.layer1(fOmf.unsqueeze(-1)))).squeeze()
          # attention = x = F.dropout(self.layer2(torch.relu(self.layer1(fOmf.unsqueeze(-1)))).squeeze(), self.opt['dropout'], training=self.training)
          attention = self.layer2(torch.relu(self.layer1(fOmf.unsqueeze(-1)))).squeeze()
        else:
          attention = fOmf
      elif self.opt['gnl_activation'] == 'identity':
        if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt[
          'wandb_epoch_list'] and self.get_evol_stats:  # not self.training:
          fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                              dst_x * dst_deginvsqrt.unsqueeze(dim=1))  # calc'd just for stats
        else:
          fOmf = torch.ones(src_deginvsqrt.shape, device=self.device)
        attention = torch.tensor([1], device=self.device) #1#torch.ones(src_deginvsqrt.shape, device=self.device)


    elif self.opt['gnl_style'] in ['att_rep_laps']:
      # get degrees
      src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)
      # calc bilinear form
      if self.opt['gnl_activation'] != 'identity':
        #cat to match indexes of sparse graph laplacian
        grad = torch.cat([src_x * src_deginvsqrt.unsqueeze(dim=1) - dst_x * dst_deginvsqrt.unsqueeze(dim=1), torch.zeros((self.n_nodes, self.in_features), device=self.device)], dim=0)
        fOmf = torch.einsum("ij,jk,ik->i", grad, self.Ws, grad)

        ag = src_x * src_deginvsqrt.unsqueeze(dim=1) + dst_x * dst_deginvsqrt.unsqueeze(dim=1)
        loop_anti_grad = 2 * self.deg_inv_sqrt.unsqueeze(dim=1) * x
        anti_grad = torch.cat([ag, loop_anti_grad], dim=0)
        anti_fOmf = torch.einsum("ij,jk,ik->i", anti_grad, self.R_Ws, anti_grad)


        # if self.opt['gnl_activation'] == "perona_malik":
        #   attention = self.pm_a**2 / (self.pm_a**2 + torch.exp(-self.pm_b**2 * fOmf))
        #   anti_attention = self.anti_pm_a**2 / (self.anti_pm_a**2 + torch.exp(-self.anti_pm_b**2 * anti_fOmf))
        # elif self.opt['gnl_activation'] == "pm_gaussian":
        #   attention = (self.pm_g**2 + 1) * torch.exp(-(fOmf-self.pm_m**2)**2/self.pm_s**2)
        #   anti_attention = (self.anti_pm_g ** 2 + 1) * torch.exp(-(anti_fOmf - self.anti_pm_m ** 2) ** 2 / self.anti_pm_s ** 2)
        if self.opt['gnl_activation'] == "pm_invsq":
          attention = self.pm_g**2 / (1 + self.pm_m**2 * fOmf**2)
          anti_attention = self.anti_pm_g**2 / (1 + self.anti_pm_m**2 * anti_fOmf**2)

      elif self.opt['gnl_activation'] == 'identity':
        if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt[
          'wandb_epoch_list'] and self.get_evol_stats:  # not self.training:
          fOmf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), self.gnl_W,
                              dst_x * dst_deginvsqrt.unsqueeze(dim=1))  # calc'd just for stats
        else:
          fOmf = torch.ones(src_deginvsqrt.shape, device=self.device)
          anti_fOmf = torch.ones(src_deginvsqrt.shape, device=self.device)

        attention = torch.tensor([1], device=self.device) #1#torch.ones(src_deginvsqrt.shape, device=self.device)
        anti_attention = torch.tensor([1], device=self.device) #1#torch.ones(src_deginvsqrt.shape, device=self.device)

      fOmf = (fOmf, anti_fOmf)
      attention = (attention, anti_attention)

    elif self.opt['gnl_style'] in ['attention_flavour']:
      fOmf = torch.einsum("ij,jk,ik->i", src_x, self.gnl_W, dst_x)
      attention = softmax(fOmf / torch.sqrt(torch.tensor([self.opt['hidden_dim']], device=self.device)), self.edge_index[self.opt['attention_norm_idx']])
      attention = make_symmetric_unordered(self.edge_index, attention)

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


  def reset_gnl_W_eigs(self, T):
    # T = int(t / self.opt['step_size'])
    #call dense W propagation matrix and set W_diag, W_U as required in the function call
    if self.time_dep_w in ["unstruct", "struct", "struct_gaus", "struct_decay"]:
      W = self.set_gnlWS_timedep(T)
    else:
      W = self.set_gnlWS()

    #set eigen_values and vectors of W
    if self.opt['gnl_W_style'] in ['GS', 'GS_Z_diag', 'cgnn', 'cgnn_Z_diag', 'loss_W_orthog', 'W_orthog_init',
                                   'householder', 'skew_sym']:
      self.W_eval, self.W_evec = self.gnl_W_D, self.gnl_W_U
    else: #ie 'Z_diag'
      self.W_eval, self.W_evec = torch.linalg.eigh(W)  # confirmed unit norm output vectors

    if self.opt['gnl_W_norm']:
      W = W / torch.abs(self.W_eval).max()

    # set progation matrix
    if self.opt['m2_W_eig'] in ['z2x', 'eye']: #if it's z2x then set W as the diag of evals
      self.gnl_W = torch.diag(self.W_eval)
    #   self.gnl_W = self.W_eval  # gofasterhadamard vector diag - possible but a faff for reporting functions
    else:
      self.gnl_W = W

    if self.opt['two_hops']:
      self.gnl_W2 = self.set_gnlWS2()

    # Omega
    if self.time_dep_omega in ["struct", "unstruct"]:
      self.Omega = self.set_gnlOmega_timedep(T)
    else:
      self.Omega = self.set_gnlOmega()

    # Sourceterm
    if self.opt['source_term'] == 'time_dep_bias':
      self.q_diag = self.q_diag_T[T]
    elif self.opt['source_term'] == 'time_dep_q':
      if self.time_dep_q in ["struct"]:
        self.gnl_Q = self.set_gnlQ_timedep(T)
      elif self.time_dep_q in ["unstruct"]:
        self.gnl_Q = self.set_gnlQ_timedep(T)

  def reset_gnl_att_rep(self, T=0):
    if self.opt['gnl_W_style'] == 'att_rep_lap_block':
      Ws, R_Ws = self.set_gnlWS()

      if self.opt['gnl_W_norm']:
        Ws_eval, Ws_evec = torch.linalg.eigh(Ws)
        Ws = Ws / torch.abs(Ws_eval).max()
        R_Ws_eval, R_Ws_evec = torch.linalg.eigh(R_Ws)
        R_Ws = R_Ws / torch.abs(R_Ws_eval).max()

    W = Ws - R_Ws
    self.W_eval, self.W_evec = torch.linalg.eigh(W)  # confirmed unit norm output vectors

    # set progation matrix
    if self.opt['m2_W_eig'] in ['z2x', 'eye']:  # if it's z2x then set W as the diag of evals
      self.gnl_W = torch.diag(self.W_eval)
    else:
      self.gnl_W = W

    self.Ws = Ws
    self.R_Ws = R_Ws

    # Omega
    if self.time_dep_omega in ["struct", "unstruct"]:
      self.Omega = self.set_gnlOmega_timedep(T)
    else:
      self.Omega = self.set_gnlOmega()

    # Sourceterm
    if self.opt['source_term'] == 'time_dep_bias':
      self.q_diag = self.q_diag_T[T]
    elif self.opt['source_term'] == 'time_dep_q':
      if self.time_dep_q in ["struct"]:
        self.gnl_Q = self.set_gnlQ_timedep(T)
      elif self.time_dep_q in ["unstruct"]:
        self.gnl_Q = self.set_gnlQ_timedep(T)

  def add_source(self, f, x):
    if self.opt['source_term'] == 'scalar':
      f = f + self.q_scalar * self.x0
    elif self.opt['source_term'] == 'fidelity':
      f = f - 0.5 * self.q_fidelity * (x - self.x0)
    elif self.opt['source_term'] == 'diag':
      f = f + self.q_diag * self.x0
    elif self.opt['source_term'] == 'bias':
      f = f + self.q_diag
    elif self.opt['source_term'] == 'time_dep_bias':
      f = f + self.q_diag
    elif self.opt['source_term'] == 'time_dep_q':
      f = f + self.gnl_Q * self.x0
    elif self.opt['source_term'] == 'none':
      pass
    return f

  def forward(self, t, x):  # t is needed when called by the integrator
    self.paths.append(x) #append initial condition of the block

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    T = int(t / self.opt['step_size'])
    if (self.time_dep_w in ["struct_gaus", "struct_decay", "unstruct"] or self.time_dep_omega in ["struct", "unstruct"] or self.time_dep_q in ["struct", "unstruct"]) and t!=0:
      self.reset_gnl_W_eigs(T)

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

        elif self.opt['gnl_style'] == 'attention_flavour':
          fOmf, attention = self.calc_dot_prod_attention(src_x, dst_x)
          # fOmf = torch.einsum("ij,jk,ik->i", src_x, self.gnl_W, dst_x)
          # attention = softmax(fOmf, self.edge_index[self.opt['attention_norm_idx']])
          # attention = make_symmetric_unordered(self.edge_index, attention)
          # src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt) #todo is it efficient to calc this every time step
          P = attention # * src_deginvsqrt * dst_deginvsqrt
          f = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], x @ self.gnl_W)
          f = f

        #general graph (GCN/GraphSage) method
        elif self.opt['gnl_style'] == 'general_graph':
          fOmf, attention = self.calc_dot_prod_attention(src_x, dst_x, T)
          src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt) #todo is it efficient to calc this every time step
          P = attention * src_deginvsqrt * dst_deginvsqrt
          #warning this seems to drag on performance - can't explain
          # del fOmf
          # del src_x
          # del dst_x
          # del src_deginvsqrt
          # del dst_deginvsqrt
          # torch.cuda.empty_cache()

          # xW = x @ self.gnl_W
          if not self.opt['gnl_measure'] == 'ones':
            pass
            # f1 = torch_sparse.spmm(self.edge_index, P / src_meas, x.shape[0], x.shape[0], xW) / 2
            # f2 = torch_sparse.spmm(self.edge_index, P / dst_meas, x.shape[0], x.shape[0], xW) / 2
            # f = f1 + f2
            # f = f - torch.diag(1 / measure) @ x @ self.Omega
          else:
            if self.opt['gnl_attention']: #todo attention only implemented for measure==ones
              P = P * self.mean_attention_0

            # if self.opt['m2_W_eig'] == 'z2x':
            #   f = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], x * self.gnl_W)
            # else:
            # print(f"devices {self.edge_index.device}, {P.device}, {x.device}, {self.gnl_W.device}")

            if self.opt['two_hops']:
              Ax = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], x)
              AAx = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], Ax)
              # f = Ax @ self.gnl_W + AAx @ W
              f = Ax @ self.gnl_W + AAx @ self.gnl_W2
            elif not self.opt['hetero_undir'] and self.opt['undir_grad_flow']: #"directed gradient flow"
              AXW = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], x @ self.gnl_W)
              row, col = self.edge_index
              edge_index_T = torch.stack([col, row], dim=0)
              AtXWt = torch_sparse.spmm(edge_index_T, P, x.shape[0], x.shape[0], x @ self.gnl_W.T)
              f = AXW + AtXWt
            else:
              f = torch_sparse.spmm(self.edge_index, P, x.shape[0], x.shape[0], x @ self.gnl_W)

            f = f - x @ self.Omega

            #   xWtilde = x @ self.gnl_W_tilde
            #   AA_ei, AA_val = torch_sparse.spspmm(self.edge_index, P, self.edge_index, P, x.shape[0], x.shape[0], x.shape[0])
            #   f = f - torch_sparse.spmm(AA_ei, AA_val, x.shape[0], x.shape[0], xWtilde) / 2

        #as per old_main.tex eq (11)
        elif self.opt['gnl_style'] == 'att_rep_laps':
          (fOmf, anti_fOmf), (attention, anti_attention) = self.calc_dot_prod_attention(src_x, dst_x, x, T)

          edges = torch.cat([self.edge_index, self.self_loops], dim=1)
          LfW = 0
          RfW = 0
          if self.opt['diffusion']:
            Ws = self.Ws
            fWs = torch.matmul(x, Ws)
            LfW = torch_sparse.spmm(edges, -self.L_0 * attention, x.shape[0], x.shape[0], fWs)
          if self.opt['repulsion']:
            R_Ws = self.R_Ws
            fRWs = torch.matmul(x, R_Ws)
            RfW = torch_sparse.spmm(edges, self.R_0 * anti_attention, x.shape[0], x.shape[0], fRWs)


          #set convex combination value alpha
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
            f = self.add_source(f, x)
        else:
          f = self.add_source(f, x)

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
          #todo need to include the batch norm and non-linearities in here too
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
              stack_stats(self) #todo move this to gnn level to make sure at the end??

          self.wandb_step += 1

      # if self.opt['greed_momentum'] and self.prev_grad:
      #   f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      #   self.prev_grad = f

    # batch norm in each feature channel - # #benchmarking GNN "https://arxiv.org/pdf/2003.00982.pdf" sections: Normalization and Residual Connect" and "C.1 Graph Regression with ZINC dataset"
    if self.opt['conv_batch_norm'] == "shared":
      f = self.batchnorm_h(f)
    elif self.opt['conv_batch_norm'] == "layerwise":
      f = self.batchnorms[T](f)

    #dampening
    # f = f - (1 - self.opt['dampen_gamma']) * x / self.opt['step_size']

    #non-linearity
    if self.opt['pointwise_nonlin']:
      # return torch.relu(f)
      return torch.relu(f)
    else:
      return f

def __repr__(self):
  return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'