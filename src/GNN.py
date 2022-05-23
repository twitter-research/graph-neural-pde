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
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

  def encoder(self, x, pos_encoding=None):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  def set_attributes(self, x):
    self.odeblock.odefunc.gnl_W = self.odeblock.odefunc.set_gnlWS()
    self.odeblock.odefunc.Omega = self.odeblock.odefunc.set_gnlOmega()

  def forward_XN(self, x):
    ###forward XN
    x = self.encoder(x, pos_encoding=None)
    if not self.opt['lie_trotter'] == 'gen_2':
      self.odeblock.set_x0(x)
      self.set_attributes(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)
    return z

  def GNN_postXN(self, z):
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]
    # Activation.
    if not self.opt['XN_no_activation']:
      z = F.relu(z)
    # fc from bottleneck
    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)
    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    return z

  def forward(self, x, pos_encoding=None):
    z = self.forward_XN(x)
    z = self.GNN_postXN(z)
    ##todo: need to implement if self.opt['m2_mlp']: from base classfor GNN_early also
    # Decode each node embedding to get node label.
    if self.opt['lie_trotter'] == 'gen_2': #if we end in label diffusion block don't need to decode to logits
      if self.opt['lt_gen2_args'][-1]['lt_block_type'] != 'label':
        z = self.m2(z)
    else:
      z = self.m2(z)

    return z