import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function


# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    self.block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblocks = nn.ModuleList(
      [self.block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor) for dummy_i in range(self.n_ode_blocks)]).to(self.device)
    self.odeblock = self.block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor).to(self.device)

    self.m2 = nn.Linear(opt['hidden_dim'], dataset.num_classes)

  def forward(self, x):
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    if self.training:
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
    return z
