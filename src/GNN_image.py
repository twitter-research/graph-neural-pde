import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from collections import namedtuple


# Define the GNN model.
class GNN_image(BaseGNN):
  def __init__(self, opt, data, num_classes, device=torch.device('cpu')):
    DataWrapper = namedtuple('DataWrapper', ['num_features'])
    dw = DataWrapper(1)
    DatasetWrapper = namedtuple('DatasetWrapper', ['data', 'num_classes'])
    dsw = DatasetWrapper(dw, num_classes)
    super(GNN_image, self).__init__(opt, dsw, device)
    self.f = set_function(opt)
    self.block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblocks = nn.ModuleList(
      [self.block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor) for dummy_i in
       range(self.n_ode_blocks)]).to(self.device)
    self.odeblock = self.block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor).to(self.device)

    self.m2 = nn.Linear(opt['im_width'] * opt['im_height'] * opt['im_chan'], num_classes)

  def forward(self, x):
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)

    self.odeblock.set_x0(x)

    if self.training:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z

  def forward_plot_T(self, x):  # the same as forward but without the decoder
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)

    self.odeblock.set_x0(x)

    if self.training:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])

    return z

  def forward_plot_path(self, x, frames):  # stitch together ODE integrations
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    z = x
    paths = [z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])]
    for f in range(frames):
      self.odeblock.set_x0(z)  # (x)
      if self.training:
        z, self.reg_states = self.odeblock(z)
      else:
        z = self.odeblock(z)
      # Activation.
      z = F.relu(z)
      # Dropout.
      z = F.dropout(z, self.opt['dropout'], training=self.training)
      path = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
      print(
        f"Total Pixel intensity of the first image: {torch.sum(z[0:self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'], :])}")
      print(f"{torch.sum(z[1:self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'], :])}")
      print(f"{torch.sum(z[2:self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'], :])}")

      paths.append(path)

    paths = torch.stack(paths, dim=1)
    return paths
