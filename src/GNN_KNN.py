import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from graph_rewiring import KNN, add_edges, edge_sampling


# Define the GNN model.
class GNN_KNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN_KNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

  def forward(self, x, pos_encoding):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      if self.opt['dataset'] == 'ogbn-arxiv':
        p = pos_encoding
      else:
        p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
        p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)
    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    # if self.training and self.odeblock.nreg > 0:
    #   z, self.reg_states = self.odeblock(x)
    # else:
    #   z = self.odeblock(x)

    if self.opt['KNN_online']:
      z = x
      for _ in range(self.opt['KNN_online_reps']):
        self.odeblock.set_x0(z)
        if self.training and self.odeblock.nreg > 0:
          z, self.reg_states = self.odeblock(z)
        else:
          z = self.odeblock(z)
        self.odeblock.odefunc.edge_index = KNN(z, self.opt)

    elif self.opt['edge_sampling_online']:
      z = x
      for _ in range(self.opt['edge_sampling_online_reps']):
        self.odeblock.set_x0(z)
        if self.training and self.odeblock.nreg > 0:
          z, self.reg_states = self.odeblock(z)
        else:
          z = self.odeblock(z)
        self.odeblock.odefunc.edge_index = add_edges(self, self.opt)
        self.odeblock.odefunc.edge_index = edge_sampling(self, z, self.opt)

    else:
      self.odeblock.set_x0(x)
      if self.training and self.odeblock.nreg > 0:
        z, self.reg_states = self.odeblock(x)
      else:
        z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # if self.opt['batch_norm']:
    #   z = self.bn_in(z)

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z

  def forward_encoder(self, x, pos_encoding):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      if self.opt['dataset'] == 'ogbn-arxiv':
        p = pos_encoding
      else:
        # p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
        p = self.mp(pos_encoding)
      x = torch.cat([x, p], dim=1)
    else:
      # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      # x = F.dropout(x, self.opt['dropout'], training=self.training)
      # x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      # x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
      x = x + self.m11(F.relu(x))
      x = x + self.m12(F.relu(x))

    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)
    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  def forward_ODE(self, x, pos_encoding):
    x = self.forward_encoder(x, pos_encoding)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    return z

# todo this was looking at rewiring online in diffusion, it runs.. but question over differentiability
def forward_online(self, x):
  # Encode each node based on its feature.
  if self.opt['use_labels']:
    y = x[:, -self.num_classes:]
    x = x[:, :-self.num_classes]

  if self.opt['beltrami']:
    p = x[:, self.num_data_features:]
    x = x[:, :self.num_data_features]
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    p = F.dropout(p, self.opt['input_dropout'], training=self.training)
    x = self.mx(x)
    p = self.mp(p)
    x = torch.cat([x, p], dim=1)
  else:
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

  if self.opt['use_mlp']:
    x = F.dropout(x, self.opt['dropout'], training=self.training)
    x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
    x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

  # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
  # if True:
  #   x = F.relu(x)
  if self.opt['use_labels']:
    x = torch.cat([x, y], dim=-1)

  if self.opt['batch_norm']:
    x = self.bn_in(x)

  # Solve the initial value problem of the ODE.
  if self.opt['augment']:
    c_aux = torch.zeros(x.shape).to(self.device)
    x = torch.cat([x, c_aux], dim=1)

  if self.opt['KNN_online']:
    z = x
    for _ in range(self.opt['KNN_online_reps']):
      self.odeblock.set_x0(z)

      if self.training and self.odeblock.nreg > 0:
        z, self.reg_states = self.odeblock(z)
      else:
        z = self.odeblock(z)

      self.odeblock.odefunc.edge_index = KNN(z, self.opt)

  else:
    self.odeblock.set_x0(x)
    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

  if self.opt['augment']:
    z = torch.split(z, x.shape[1] // 2, dim=1)[0]

  # if self.opt['batch_norm']:
  #   z = self.bn_in(z)

  # Activation.
  z = F.relu(z)

  if self.opt['fc_out']:
    z = self.fc(z)
    z = F.relu(z)

  # Dropout.
  z = F.dropout(z, self.opt['dropout'], training=self.training)

  # Decode each node embedding to get node label.
  z = self.m2(z)
  return z
