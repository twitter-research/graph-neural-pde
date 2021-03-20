from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
import torch_sparse
from torch_geometric.utils import get_laplacian
import numpy as np

class ConstantODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)

    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    if opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()


  def add_random_edges(self):
    #todo check if theres a pygeometric function for this

    # M = self.opt["M_nodes"]
    M = int(self.num_nodes * (1/(1 - (1 - self.opt['att_samp_pct'])) - 1))

    with torch.no_grad():
      new_edges = np.random.choice(self.num_nodes, size=(2,M), replace=True, p=None)
      new_edges = torch.tensor(new_edges)
      cat = torch.cat([self.data_edge_index, new_edges],dim=1)
      no_repeats = torch.unique(cat, sorted=False, return_inverse=False,
                                return_counts=False, dim=0)
      self.data_edge_index = no_repeats

  def add_khop_edges(self, k):
    n = self.num_nodes
    # do k_hop
    for i in range(k):
      new_edges, new_weights = \
        torch_sparse.spspmm(self.odefunc.edge_index, self.odefunc.edge_weight,
                            self.odefunc.edge_index, self.odefunc.edge_weight, n, n, n, coalesced=False)
    self.edge_weight = 0.5 * self.edge_weight + 0.5 * new_weights
    cat = torch.cat([self.data_edge_index, new_edges], dim=1)
    self.edge_index = torch.unique(cat, sorted=False, return_inverse=False,
                                   return_counts=False, dim=0)
    # threshold
    # normalise

  # self.odefunc.edge_index, self.odefunc.edge_weight =
  # get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  # num_nodes = maybe_num_nodes(edge_index, num_nodes)


  def forward(self, x):
    t = self.t.type_as(x)

    if self.training:
      if self.opt['new_edges'] == 'random':
        self.add_random_edges()
      elif self.opt['new_edges'] == 'k_hop':
        self.add_khop_edges(k=2)
      elif self.opt['new_edges'] == 'random_walk' and self.odefunc.attention_weights is not None:
        self.add_rw_edges()



    attention_weights = self.get_attention_weights(x)
    # create attention mask
    if self.training:
      with torch.no_grad():
        mean_att = attention_weights.mean(dim=1, keepdim=False)
        if self.opt['use_flux']:
          src_features = x[self.data_edge_index[0, :], :]
          dst_features = x[self.data_edge_index[1, :], :]
          delta = torch.linalg.norm(src_features-dst_features, dim=1)
          mean_att = mean_att * delta
        threshold = torch.quantile(mean_att, 1-self.opt['att_samp_pct'])
        mask = mean_att > threshold
        self.odefunc.edge_index = self.data_edge_index[:, mask.T]
        sampled_attention_weights = self.renormalise_attention(mean_att[mask])
        print('retaining {} of {} edges'.format(self.odefunc.edge_index.shape[1], self.data_edge_index.shape[1]))
        self.odefunc.attention_weights = sampled_attention_weights
    else:
      self.odefunc.edge_index = self.data_edge_index
      self.odefunc.attention_weights = attention_weights.mean(dim=1, keepdim=False)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights




    integrator = self.train_integrator if self.training else self.test_integrator
    
    reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      return z, reg_states
    else: 
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
