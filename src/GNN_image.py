import torch
from torch import nn
import torch.nn.functional as F
from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_dorsey_attention import ODEFuncDorseyAtt
from function_laplacian_diffusion import LaplacianODEFunc
from sde import SDEFunc, SDEblock
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from block_mixed import MixedODEblock
from base_classes import BaseGNN
from model_configurations import set_block, set_function

#TODO review the implementation for the 3 types of forward pass below, same augmentation/dropout/activation/encoder/decoder etc??

# Define the GNN model.
class GNN_image(BaseGNN):
  def __init__(self, opt, num_features, num_nodes, num_classes, edge_index, edge_attr=None, device=torch.device('cpu')):
    super(GNN_image, self).__init__(opt, num_features, device)
    self.f = set_function(opt)
    self.block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    # self.odeblocks = nn.ModuleList(
    #   [self.block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor) for dummy_i in range(self.n_ode_blocks)]).to(self.device)
    self.odeblock = self.block(self.f, self.regularization_fns, opt, num_nodes, edge_index, edge_attr, device, t=time_tensor).to(self.device)

    self.batchnorm = nn.BatchNorm2d(num_features=opt['im_chan'])

    self.m2 = nn.Linear(opt['im_width'] * opt['im_height'] * opt['im_chan'], num_classes)

  def forward(self, x):
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    # x = self.m1(x) #no encoding for image viz
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)

    # Solve the initial value problem of the ODE.
    # if self.opt['augment']: #no augmenting for image viz
    #   c_aux = torch.zeros(x.shape).to(self.device)
    #   x = torch.cat([x, c_aux], dim=1)
    x = self.batchnorm(x)
    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    # if self.opt['augment']: #no augmenting for image viz
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z

  def forward_plot_T(self, x): #the same as forward but without the decoder
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    # x = self.m1(x) #no encoding for image viz
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)

    # Solve the initial value problem of the ODE.
    # if self.opt['augment']: #no augmenting for image viz
    #   c_aux = torch.zeros(x.shape).to(self.device)
    #   x = torch.cat([x, c_aux], dim=1)
    x = self.batchnorm(x)
    self.odeblock.set_x0(x)

    if self.training:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    # if self.opt['augment']: #no augmenting for image viz
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])

    return z

  def forward_plot_path(self, x, frames): #stitch together ODE integrations
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    # x = self.m1(x) #no encoding for image viz
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)

    # Solve the initial value problem of the ODE.
    # if self.opt['augment']: #no augmenting for image viz
    #   c_aux = torch.zeros(x.shape).to(self.device)
    #   x = torch.cat([x, c_aux], dim=1)

    x = self.batchnorm(x)

    z = x
    paths = [z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])]
    for f in range(frames):
      self.odeblock.set_x0(z) #(x)
      if self.training:
        z, self.reg_states = self.odeblock(z)
      else:
        z = self.odeblock(z)
      # Activation.
      z = F.relu(z)
      # Dropout.
      z = F.dropout(z, self.opt['dropout'], training=self.training)
      path = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
      print(f"Total Pixel intensity of the first image: {torch.sum(z[0:self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'],:])}")
      paths.append(path)

    paths = torch.stack(paths,dim=1)
    return paths