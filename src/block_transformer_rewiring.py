import torch
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
from utils import get_rw_adj
from torch_scatter import scatter
import numpy as np
import torch_sparse
from torch_geometric.utils import remove_self_loops

class RewireAttODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(RewireAttODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)
    assert opt['att_samp_pct'] > 0 and opt['att_samp_pct'] <= 1, "attention sampling threshold must be in (0,1]"
    self.opt = opt
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    self.num_nodes = data.num_nodes
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.data_edge_index = edge_index.to(device)
    self.odefunc.edge_index = edge_index.to(device)  # this will be changed by attention scores
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    if opt['function'] not in {'GAT', 'transformer'}:
      self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device, edge_weights=self.odefunc.edge_weight).to(device)

  def get_attention_weights(self, x):
    if self.opt['function'] not in {'GAT', 'transformer'}:
      attention, values = self.multihead_att_layer(x, self.data_edge_index)
    else:
      attention, values = self.odefunc.multihead_att_layer(x, self.data_edge_index)
    return attention

  def renormalise_attention(self, attention):
    index = self.odefunc.edge_index[self.opt['attention_norm_idx']]
    att_sums = scatter(attention, index, dim=0, dim_size=self.num_nodes, reduce='sum')[index]
    return attention / (att_sums + 1e-16)


  def add_random_edges(self):
    # M = self.opt["M_nodes"]
    M = int(self.num_nodes * (1/(1 - (1 - self.opt['att_samp_pct'])) - 1))

    with torch.no_grad():
      new_edges = np.random.choice(self.num_nodes, size=(2,M), replace=True, p=None)
      new_edges = torch.tensor(new_edges)
      #todo check if should be using coalesce insted of unique
      #eg https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/two_hop.html#TwoHop
      cat = torch.cat([self.data_edge_index, new_edges],dim=1)
      no_repeats = torch.unique(cat, sorted=False, return_inverse=False,
                                return_counts=False, dim=0)
      self.data_edge_index = no_repeats


  def add_khop_edges(self, k=2, rm_self_loops=True):
    n = self.num_nodes
    for i in range(k-1):
      new_edges, new_weights = torch_sparse.spspmm(self.odefunc.edge_index, self.odefunc.edge_weight,
                            self.odefunc.edge_index, self.odefunc.edge_weight, n, n, n, coalesced=True)

      new_edges, new_weights = remove_self_loops(new_edges, new_weights)

      A1 = torch.sparse_coo_tensor(self.odefunc.edge_index, self.odefunc.edge_weight, (n, n)).coalesce()
      A2 = torch.sparse_coo_tensor(new_edges, new_weights, (n, n)).coalesce()

    if self.opt['threshold_type'] == 'topk_adj':
      S_hat = (0.5 * A1 + 0.5 * A2).coalesce()
      self.data_edge_index = S_hat.indices()
      self.odefunc.attention_weights = S_hat.values()

    elif self.opt['threshold_type'] == 'addD_rvR':
      AN = A2
      npA1idx = A1.indices().numpy().T
      npANidx = AN.indices().numpy().T

      # https: // stackoverflow.com / questions / 16970982 / find - unique - rows - in -numpy - array
      # A1_rows = npA1idx.view([('', npA1idx.dtype)] * npA1idx.shape[0])
      # AN_rows = npANidx.view([('', npANidx.dtype)] * npANidx.shape[0])

      # data = npA1idx
      # ncols = data.shape[1]
      # dtype = data.dtype.descr * ncols
      # struct = data.view(dtype)
      #
      # uniq = np.unique(struct)
      # uniq = uniq.view(data.dtype).reshape(-1, ncols)
      # removed_mask = np.in1d(A1_rows, AN_rows, assume_unique=True, invert=True)
      # added_mask = np.in1d(AN_rows, A1_rows, assume_unique=True, invert=True)

      # combined = torch.cat((A1.indices(), AN.indices()), dim=1)
      # uniques, counts = combined.unique(return_counts=True, dim=1)
      # difference = uniques[counts == 1]
      # intersection = uniques[counts > 1]

      A1_rows = np.ascontiguousarray(npA1idx).view(np.dtype((np.void, npA1idx.dtype.itemsize * npA1idx.shape[1])))
      AN_rows = np.ascontiguousarray(npANidx).view(np.dtype((np.void, npANidx.dtype.itemsize * npANidx.shape[1])))
      #todo use jax.numpy.in1d to do on GPU
      removed_mask = np.in1d(A1_rows, AN_rows, assume_unique=True, invert=True)
      added_mask = np.in1d(AN_rows, A1_rows, assume_unique=True, invert=True)

      assert len(A1_rows)+added_mask.sum()-removed_mask.sum()-len(AN_rows)==0

      threshold = torch.quantile(AN.values()[added_mask], 1 - self.opt['rw_addD'])
      threshold_mask = AN.values()[added_mask] > threshold

      add_edges = npANidx[added_mask,:][threshold_mask,:]
      add_values = AN.values()[added_mask][threshold_mask]

      combined_edges = torch.cat((self.odefunc.edge_index, torch.from_numpy(add_edges).T), dim=1)
      combined_values = torch.cat((self.odefunc.edge_weight, add_values))

      self.data_edge_index = combined_edges
      self.odefunc.attention_weights = combined_values

  # def add_rw_edges(self): #NOT COMPLETE
  #   # function to sample M random walks rather than densifying Adjacency
  #   # https: // github.com / rusty1s / pytorch_sparse / blob / master / torch_sparse / sample.py
  #   # def sample(src: SparseTensor, num_neighbors: int,
  #   #            subset: Optional[torch.Tensor] = None) -> torch.Tensor:
  #   M = int(self.num_nodes * (1/(1 - (1 - self.opt['att_samp_pct'])) - 1))
  #   with torch.no_grad():
  #     M_start = np.random.choice(self.num_nodes, size=(M), replace=True, p=None)
  #     scale = 3.0
  #     L = np.abs(np.random.normal(loc=0, scale=scale, size=(M)))
  #     attention_weights = self.odefunc.attention_weights
  #     M_end = torch.zeros(M)
  #     for m, m_start in enumerate(M_start):
  #       fuel = L[m]
  #       while fuel > 0:
  #         current_node_mask = self.data_edge_index[0,:] == m_start
  #         p = attention_weights * current_node_mask
  #         m_start = np.random.choice(len(p), size=(M), replace=True, p=p)
  #         fuel -= 1 #written this way in case change cost of path length from 1
  #       M_end[m] = m_start
  #     # keep going until all steps taken
  #     # L[m] -= 1

  def densify_edges(self):
    if self.opt['new_edges'] == 'random':
      self.add_random_edges()
    elif self.opt['new_edges'] == 'random_walk':
      self.add_rw_edges()
    elif self.opt['new_edges'] == 'k_hop_lap':
      pass
    elif self.opt['new_edges'] == 'k_hop_att':
      self.add_khop_edges(k=2)

  def threshold_edges(self, x):
    # get mean attention
    # i) sparsify on S_hat
    if self.opt['sparsify'] == 'S_hat':
      attention_weights = self.odefunc.attention_weights
      mean_att = attention_weights

    # ii) sparsify on recalced attention
    elif self.opt['sparsify'] == 'recalc_att':
      attention_weights = self.get_attention_weights(x)
      mean_att = attention_weights.mean(dim=1, keepdim=False)

    if self.opt['use_flux']:
      src_features = x[self.data_edge_index[0, :], :]
      dst_features = x[self.data_edge_index[1, :], :]
      delta = torch.linalg.norm(src_features - dst_features, dim=1)
      mean_att = mean_att * delta

    unique_att = torch.unique(mean_att, sorted=False, return_inverse=False, return_counts=False, dim=0)  #just for the test where threshold catches all edges
    print(f"mean_att {mean_att.shape}, unqiue atts: {unique_att.shape}")
    # threshold
    threshold = torch.quantile(mean_att, 1 - self.opt['att_samp_pct'])
    mask = mean_att > threshold
    self.odefunc.edge_index = self.data_edge_index[:, mask.T]
    sampled_attention_weights = self.renormalise_attention(mean_att[mask])
    print('retaining {} of {} edges'.format(self.odefunc.edge_index.shape[1], self.data_edge_index.shape[1]))
    self.data_edge_index = self.data_edge_index[:, mask.T]
    self.odefunc.edge_weight = sampled_attention_weights
    self.odefunc.attention_weights = sampled_attention_weights

  def forward(self, x):
    t = self.t.type_as(x)

    if self.training:
      with torch.no_grad():
        #calc attentions for transition matrix
        attention_weights = self.get_attention_weights(x)
        self.odefunc.attention_weights = attention_weights.mean(dim=1, keepdim=False)

        # Densify and threshold attention weights
        self.densify_edges()
        self.threshold_edges(x)

    self.odefunc.edge_index = self.data_edge_index
    attention_weights = self.get_attention_weights(x)
    mean_att = attention_weights.mean(dim=1, keepdim=False)
    self.odefunc.edge_weight = mean_att
    self.odefunc.attention_weights = mean_att

    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator
    reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))
    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple(st[1] for st in state_dt[1:])
      return z, reg_states
    else:
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
