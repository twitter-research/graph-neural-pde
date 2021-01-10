import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from function_GAT_attention import ODEFuncAtt
from sklearn.preprocessing import normalize
import scipy
adjoint = True
if adjoint:
  from torchsde import sdeint_adjoint as sdeint
else:
  from torchsde import sdeint

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class SDEFunc(MessagePassing):
  noise_type = 'diagonal'
  sde_type = 'ito'

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device, brownian_size=32):
    super(SDEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.alpha = opt['alpha']
    self.num_nodes = data.num_nodes
    self.beta = 1
    self.improved = False
    self.add_self_loops = True
    self.adj = self.get_rw_adj(data, self_loops=True)
    self.x0 = None
    self.nfe = 0
    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.gamma_sc = nn.Parameter(torch.ones(1))
    self.brownian_size = brownian_size
    self.gamma = nn.Parameter(torch.ones(self.num_nodes))
    # self.sigma = torch.nn.Linear(out_features,
    #                              out_features * brownian_size)
    self.sigma = torch.nn.Linear(out_features,
                                 out_features)



  def coo2tensor(self, coo):
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    values = coo.data
    v = torch.FloatTensor(values)
    shape = coo.shape
    print('adjacency matrix generated with shape {}'.format(shape))
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)

  def get_rw_adj(self, data, self_loops=False):
    coo = to_scipy_sparse_matrix(data.edge_index, data.edge_attr)
    if self_loops:
      coo = coo + scipy.sparse.eye(data.num_nodes)
    normed_csc = normalize(coo, norm='l1', axis=0)
    return self.coo2tensor(normed_csc.tocoo())

  def f(self, t, x):
    self.nfe += 1
    ax = torch.spmm(self.adj, x)  # [n_nodes, 2 * hidden]
    f = self.alpha_sc * (ax - x) + self.beta_sc * self.x0
    return f

  def g(self, t, x):
    # return self.sigma(x).view(self.num_nodes,
    #                           self.out_features,
    #                           self.brownian_size)
    # return self.sigma(x).view(self.num_nodes,
    #                         self.out_features)
    # return torch.mm(torch.diag_embed(self.gamma), x)
    return self.gamma_sc * x


class SDEblock(nn.Module):
  def __init__(self, sdefunc, opt, t=torch.tensor([0, 1])):
    super(SDEblock, self).__init__()
    self.t = t
    self.odefunc = sdefunc  # called odefunc instead of sdefunc for upstream compatability
    if opt['adjoint']:
      from torchsde import sdeint_adjoint as sdeint
    else:
      from torchsde import sdeint
    self.integrator = sdeint

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()

  def forward(self, x):
    t = self.t.type_as(x)
    z = self.integrator(self.odefunc, x, t, rtol=1e-2)[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"