import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from utils import project_paths_label_space#, project_paths_logit_space
from torch_geometric.utils import contains_self_loops
from greed_reporting_fcts import test

# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
    self.odeblock.odefunc.GNN_postXN = self.GNN_postXN
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
      x = self.m1(  x)

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
      self.odeblock.odefunc.set_x_0(x) #this x is actually z needed for the linear models
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
    elif self.opt['function'] in ['greed_non_linear', 'greed_lie_trotter']:
      # self.odeblock.odefunc.paths = []
      if self.opt['gnl_style'] == 'scaled_dot':
        self.odeblock.odefunc.Omega = self.odeblock.odefunc.set_scaled_dot_omega()
      elif self.opt['gnl_style'] == 'general_graph':
        W = self.odeblock.odefunc.set_gnlWS()

        if self.opt['gnl_W_style'] == 'GS_Z_diag':
          self.W_eval, self.W_evec = self.odeblock.odefunc.gnl_W_D, self.odeblock.odefunc.V_hat
        else:
          self.W_eval, self.W_evec = torch.linalg.eigh(W) #confirmed unit norm output vectors

        self.odeblock.odefunc.W_eval, self.odeblock.odefunc.W_evec = self.W_eval, self.W_evec

        if self.opt['gnl_W_norm']: #need to do this at the GNN level as called once in the forward call
          # W_eval, W_evec = torch.linalg.eigh(W)
          W = W / torch.abs(self.W_eval).max()

        if self.opt['gnl_W_style'] == 'Z_diag':
          self.odeblock.odefunc.gnl_W = torch.diag(self.W_eval)
        elif self.opt['gnl_W_style'] == 'GS_Z_diag':
          self.odeblock.odefunc.gnl_W = torch.diag(self.W_eval)
        else:
          self.odeblock.odefunc.gnl_W = W

        self.odeblock.odefunc.Omega = self.odeblock.odefunc.set_gnlOmega()
        if self.opt['two_hops']:
          self.odeblock.odefunc.gnl_W_tilde = self.odeblock.odefunc.set_gnlWS()
        if self.opt['gnl_attention']:
          self.odeblock.odefunc.set_M0()

      elif self.opt['gnl_style'] == 'att_rep_laps': #contains_self_loops(self.odeblock.odefunc.edge_index)
        if self.opt['gnl_W_style'] == 'att_rep_lap_block':
          Ws, R_Ws = self.odeblock.odefunc.set_gnlWS()
          if self.opt['gnl_W_norm']: #need to do this at the GNN level as called once in the forward call
            Ws_eval, Ws_evec = torch.linalg.eigh(Ws)
            Ws = Ws / torch.abs(Ws_eval).max()
            R_Ws_eval, R_Ws_evec = torch.linalg.eigh(R_Ws)
            R_Ws= R_Ws / torch.abs(R_Ws_eval).max()
          self.odeblock.odefunc.Ws = Ws
          self.odeblock.odefunc.R_Ws = R_Ws
          self.odeblock.odefunc.gnl_W = Ws - R_Ws

        elif self.opt['gnl_W_style'] == 'sum':
          pass #just testing
          # self.odeblock.odefunc.Ws = self.odeblock.odefunc.set_gnlWS()
          # self.odeblock.odefunc.R_Ws = self.odeblock.odefunc.set_gnlWS()
          # self.odeblock.odefunc.gnl_W = self.odeblock.odefunc.set_gnlWS()

        if self.opt['diffusion']:
          self.odeblock.odefunc.set_L0()
        if self.opt['repulsion']:
          self.odeblock.odefunc.set_R0()

  def set_W_spec(self):
    ### function to calc and control W spec ech epoch
    pass


  def forward_XN(self, x):
    ###forward XN
    x = self.encoder(x, pos_encoding=None)

    self.odeblock.odefunc.paths = []
    if not self.opt['lie_trotter'] == 'gen_2': #do for gen2 in the odeblock as copy over initial conditions
      self.odeblock.set_x0(x)
      self.set_attributes(x)

    if self.opt['m1_W_eig']:
      #project x0 into eigen space of W_eval
      x = x @ self.W_evec # X(t) = Z(t)U_{W}.T  iff X(t)U_{W} = Z(t)  # sorry switching notion between math and code

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)
    return z

  def GNN_postXN(self, z):
    if self.opt['augment']:
      z = torch.split(z, z.shape[1] // 2, dim=1)[0]
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
    self.odeblock.odefunc.paths.append(z)

    z = self.GNN_postXN(z)

    ##todo: need to implement if self.opt['m2_mlp']: from base GNN class for GNN_early also
    # Decode each node embedding to get node label.
    if self.opt['m2_aug']:
      return z[:, self.hidden_dim - self.num_classes:]

    if self.opt['m2_W_eig'] =='x2z':
      z = z @ self.W_evec # X(t) = Z(t)U_{W}.T  iff X(t)U_{W} = Z(t)  # sorry switching notion between math and code
      z = self.m2(z)
      return z
    elif self.opt['m2_W_eig'] == 'z2x':
      z = z @ self.W_evec.T # X(t) = Z(t)U_{W}.T  iff X(t)U_{W} = Z(t)  # sorry switching notion between math and code
      z = self.m2(z)
      return z

    if self.opt['m3_path_dep']:
      paths = torch.stack(self.odeblock.odefunc.paths, axis=-1)
      if self.opt['m3_space'] == 'label':
        paths = project_paths_label_space(self.m2, paths)
      return self.m3(paths.reshape(self.num_nodes, -1))

    # if self.opt['m3_best_path_dep']:
    #   paths = torch.stack(self.odeblock.odefunc.paths, axis=-1)
    #   path_preds = project_paths_logit_space(self.m2, paths)
    #   # convert paths to preds and then accuracies
    #   # identify time step that each group acheived the best test performance using torch.argmax
    #   # input the path dependent prediction with a "STRONG" hint of which time step it should consider via "attention"
    #   return self.m3(paths.reshape(self.num_nodes, -1))
    # def predict(self, z):
    #   z = self.GNN_postXN(z)
    #   logits = self.GNN_m2(z)
    #   pred = logits.max(1)[1]
    #   return logits, pred

    if self.opt['lie_trotter'] == 'gen_2': #if we end in label diffusion block don't need to decode to logits
      if self.opt['lt_gen2_args'][-1]['lt_block_type'] != 'label':
        z = self.m2(z)
    else:
      z = self.m2(z)


    return z