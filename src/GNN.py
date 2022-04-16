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

    self.odeblock.odefunc.GNN_m2 = self.m2

  def encoder(self, x, pos_encoding=None):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
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

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  def set_attributes(self, x):
    if self.opt['function'] in ['greed_linear', 'greed_linear_homo', 'greed_linear_hetero']:
      self.odeblock.odefunc.set_x_0(x) #this x is actually z
      self.odeblock.odefunc.set_tau_0()
      if self.opt['function'] == 'greed_linear_homo':
        self.odeblock.odefunc.set_L0()
      if self.opt['function'] == 'greed_linear_hetero':
        if self.opt['diffusion']:
          self.odeblock.odefunc.set_L0()
          self.odeblock.odefunc.Ws = self.odeblock.odefunc.set_WS(x)
        if self.opt['repulsion']:
          self.odeblock.odefunc.set_R0()
          self.odeblock.odefunc.R_Ws = self.odeblock.odefunc.set_WS(x)
    if self.opt['function'] == "greed_non_linear" and self.opt['gnl_style'] == 'general_graph':
          self.odeblock.odefunc.gnl_W = self.odeblock.odefunc.set_gnlWS()

  def forward_XN(self, x):
    ###forward XN
    x = self.encoder(x, pos_encoding=None)
    self.odeblock.set_x0(x)
    self.set_attributes(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    if not self.opt['XN_no_activation']:
      z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    return z

  def forward(self, x, pos_encoding=None):
    # # Encode each node based on its feature.
    # if self.opt['use_labels']:
    #   y = x[:, -self.num_classes:]
    #   x = x[:, :-self.num_classes]
    #
    # if self.opt['beltrami']:
    #   x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    #   x = self.mx(x)
    #   p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
    #   p = self.mp(p)
    #   x = torch.cat([x, p], dim=1)
    # else:
    #   x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    #   x = self.m1(x)
    #
    # if self.opt['use_mlp']:
    #   x = F.dropout(x, self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    #
    # if self.opt['use_labels']:
    #   x = torch.cat([x, y], dim=-1)
    #
    # if self.opt['batch_norm']:
    #   x = self.bn_in(x)
    #
    # # Solve the initial value problem of the ODE.
    # if self.opt['augment']:
    #   c_aux = torch.zeros(x.shape).to(self.device)
    #   x = torch.cat([x, c_aux], dim=1)

    # ###forward_XN
    # x = self.encoder(x, pos_encoding=None)
    # self.odeblock.set_x0(x)
    #
    # if self.opt['function'] in ['greed_linear', 'greed_linear_homo', 'greed_linear_hetero']:
    #   self.odeblock.odefunc.set_x_0(x) #this x is actually z
    #   self.odeblock.odefunc.set_tau_0()
    #   if self.opt['function'] == 'greed_linear_homo':
    #     self.odeblock.odefunc.set_L0()
    #   if self.opt['function'] == 'greed_linear_hetero':
    #     if self.opt['diffusion']:
    #       self.odeblock.odefunc.set_L0()
    #       self.odeblock.odefunc.Ws = self.odeblock.odefunc.set_WS(x)
    #     if self.opt['repulsion']:
    #       self.odeblock.odefunc.set_R0()
    #       self.odeblock.odefunc.R_Ws = self.odeblock.odefunc.set_WS(x)
    #
    # if self.training and self.odeblock.nreg > 0:
    #   z, self.reg_states = self.odeblock(x)
    # else:
    #   z = self.odeblock(x)
    #
    # if self.opt['augment']:
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]
    #
    # # Activation.
    # z = F.relu(z)
    #
    # if self.opt['fc_out']:
    #   z = self.fc(z)
    #   z = F.relu(z)
    #
    # # Dropout.
    # z = F.dropout(z, self.opt['dropout'], training=self.training)

    if self.opt['gnl_thresholding']:
      z = x
      self.set_attributes(z)
      z = self.encoder(z, pos_encoding=None)
      for _ in range(self.opt['gnl_thresholding_reps']):
        self.odeblock.set_x0(z)
        #run evolution
        if self.training and self.odeblock.nreg > 0:
          z, self.reg_states = self.odeblock(z)
        else:
          z = self.odeblock(z)

        #predict
        if not self.opt['XN_no_activation']:
          z = F.relu(z)
        if self.opt['fc_out']:
          z = self.fc(z)
          z = F.relu(z)
        logits = self.m2(z)
        pred = logits.max(1)[1]

        #threshold label space
        Ek = F.one_hot(pred, num_classes=self.num_classes)
        #pseudo inverse
        P = self.m2.weight
        #https://pytorch.org/docs/stable/generated/torch.matrix_rank.html
        b = self.m2.bias
        P_dagg = torch.linalg.pinv(P).T  #sometimes get RuntimeError: svd_cpu: the updating process of SBDSDC did not converge (error: 4)
        z = (Ek - b.unsqueeze(0)) @ P_dagg + z @ (torch.eye(self.hidden_dim, device=self.device) - P_dagg.T @ P).T
    else:
      z = self.forward_XN(x)

    ##todo: need to implement if self.opt['m2_mlp']: from base classfor GNN_early also
    # Decode each node embedding to get node label.
    z = self.m2(z)

    return z
