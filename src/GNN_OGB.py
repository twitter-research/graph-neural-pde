import torch
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from function_OGB import OGBFunc
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def get_opt():
  opt = {}
  opt['block'] = 'constant'
  opt['hidden_dim'] = 256
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'adam'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 1
  opt['time'] = 4
  opt['num_nodes'] = 169343
  opt['epoch'] = 500
  opt['augment'] = False
  opt['adjoint'] = True
  opt['add_source'] = False
  opt['method'] = 'rk4'
  opt['adjoint_method'] = 'rk4'
  opt["kinetic_energy"] = None
  opt["jacobian_norm2"] = None
  opt["total_deriv"] = None
  opt["directional_penalty"] = None
  opt['tol_scale'] = 1
  return opt


# Define the GNN model.
class GNN_OGB(BaseGNN):
  def __init__(self, opt, dataset, adj_t, device=torch.device('cpu')):
    super(GNN_OGB, self).__init__(opt, dataset, device)
    self.f = OGBFunc
    self.block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    dataset.data.num_nodes = dataset.data.num_nodes[0]
    self.odeblock = self.block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
    self.odeblock.odefunc.adj = gcn_norm(adj_t, num_nodes=opt['num_nodes'])


  def reset_parameters(self):
    pass

  def forward(self, x, adj):
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z.log_softmax(dim=-1)
