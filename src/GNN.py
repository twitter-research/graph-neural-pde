import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter_add
from torch_geometric.utils import contains_self_loops
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from utils import rayleigh_quotient #project_paths_label_space#, project_paths_logit_space
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

    if opt['post_proc'] in ['node', 'neighbour']:
      # self.post_theta = Parameter(torch.Tensor(opt['hidden_dim'], opt['hidden_dim']))
      self.post_theta = Parameter(torch.eye(opt['hidden_dim']))

    # self.reset_parameters()

  def reset_parameters(self):
    if self.opt['post_proc'] in ['node', 'neighbour']:
      glorot(self.post_theta)

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
      if self.opt['dataset'] == "ZINC":
        x = self.m1(x.type(torch.LongTensor).to(self.device)).squeeze()
      else:
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
      elif self.opt['gnl_style'] in ['general_graph', 'attention_flavour']:
        self.odeblock.odefunc.reset_gnl_W_eigs(T=0)

        # if self.opt['two_hops']:
        #   self.odeblock.odefunc.gnl_W_tilde = self.odeblock.odefunc.set_gnlWS()
        if self.opt['gnl_attention']:
          self.odeblock.odefunc.set_M0()

      elif self.opt['gnl_style'] == 'att_rep_laps': #contains_self_loops(self.odeblock.odefunc.edge_index)
        self.odeblock.odefunc.reset_gnl_att_rep(T=0)

        # if self.opt['gnl_W_style'] == 'att_rep_lap_block':
        #   Ws, R_Ws = self.odeblock.odefunc.set_gnlWS()
        #   if self.opt['gnl_W_norm']: #need to do this at the GNN level as called once in the forward call
        #     Ws_eval, Ws_evec = torch.linalg.eigh(Ws)
        #     Ws = Ws / torch.abs(Ws_eval).max()
        #     R_Ws_eval, R_Ws_evec = torch.linalg.eigh(R_Ws)
        #     R_Ws= R_Ws / torch.abs(R_Ws_eval).max()
        #   self.odeblock.odefunc.Ws = Ws
        #   self.odeblock.odefunc.R_Ws = R_Ws
        #   self.odeblock.odefunc.gnl_W = Ws - R_Ws

        # elif self.opt['gnl_W_style'] == 'sum':
        #   pass #just testing
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
    # x = x / (torch.norm(x, p=2, dim=1).unsqueeze(-1))


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

    if self.opt['post_proc'] in ['node', 'neighbour']:
      z = self.post_processing(z)

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

  def square_p(self, x):
    x = (x + torch.sqrt(x ** 2 + 4)) / 2
    return x

  def post_processing(self, z):
    if self.opt['post_proc'] == 'node':
      # p = (torch.norm(z, p=1, dim=1).unsqueeze(-1) / (0.01 + torch.abs(z))) * torch.exp(z @ self.post_theta) / torch.exp(z @ self.post_theta).sum(dim=1).unsqueeze(-1)
      p = (torch.norm(z, p=1, dim=1).unsqueeze(-1) / (0.01 + torch.abs(z))) * self.square_p(z @ self.post_theta) / self.square_p(z @ self.post_theta).sum(dim=1).unsqueeze(-1)
    elif self.opt['post_proc'] == 'neighbour':
      theta_proj = self.square_p(z @ self.post_theta) / self.square_p(z @ self.post_theta).sum(dim=1).unsqueeze(-1)
      src_tp, dst_tp = self.odeblock.odefunc.get_src_dst(theta_proj)
      tp_sum = scatter_add(src_tp, self.odeblock.odefunc.edge_index[0], dim=0, dim_size=self.num_nodes)
      p = (torch.norm(z, p=1, dim=1).unsqueeze(-1) / (0.01 + torch.abs(z))) * tp_sum

    return p * z

  def forward(self, x, pos_encoding=None):
    z = self.forward_XN(x)
    self.odeblock.odefunc.paths.append(z)

    z = self.GNN_postXN(z) #non-linearity / drop-out / augmentation

    ##todo: need to implement if self.opt['m2_mlp']: from base GNN class for GNN_early also
    # Decode each node embedding to get node label.
    if self.opt['m2_aug']:
      return z[:, self.hidden_dim - self.num_classes:]

    if self.opt['m2_W_eig'] =='x2z':
      z = z @ self.odeblock.odefunc.W_evec # X(t) = Z(t)U_{W}.T  iff X(t)U_{W} = Z(t)  # sorry switching z/x notion between math and code
    elif self.opt['m2_W_eig'] == 'z2x':
      z = z @ self.odeblock.odefunc.W_evec.T # X(t) = Z(t)U_{W}.T  iff X(t)U_{W} = Z(t)  # sorry switching z/x notion between math and code
    elif self.opt['m2_W_eig'] == 'eye':
      pass

    paths = self.odeblock.odefunc.paths

    if self.opt['path_dep_norm'] == 'nodewise':
      paths = [F.normalize(z_t, dim=1) for z_t in paths]
    elif self.opt['path_dep_norm'] == 'rayleigh':
      # paths = [rayleigh_quotient(self.odeblock.odefunc.edge_index, self.num_nodes, z_t) for z_t in paths]
      paths = [z_t / torch.pow(torch.norm(z_t, p="fro"), 2) for z_t in paths]
    elif self.opt['path_dep_norm'] == 'rayleigh1':
      paths = [z_t / torch.norm(z_t, p="fro") for z_t in paths]
    elif self.opt['path_dep_norm'] == 'z_cat_normed_z':
      paths = [torch.cat((z_t, F.normalize(z_t, dim=1)),dim=1) for z_t in paths]

    if self.opt['m3_path_dep'] == 'feature_jk':
      paths = torch.cat(paths, axis=-1)
      return self.m3(paths)

    elif self.opt['m3_path_dep'] == 'label_jk':
      if self.opt['path_dep_norm'] == 'z_cat_normed_z':
        label_paths_list = [self.alpha_z * self.m2(p[:, :self.opt['hidden_dim']]) + (1 - self.alpha_z) * self.m2_concat(p[:, self.opt['hidden_dim']:]) for p in paths]
      else:
        label_paths_list = [self.m2(p) for p in paths]
      label_paths = torch.cat(label_paths_list, axis=-1)
      return self.m3(label_paths)

    elif self.opt['m3_path_dep'] == 'label_att':
      if self.opt['path_dep_norm'] == 'z_cat_normed_z':
        label_paths_list = [self.alpha_z * self.m2(p[:, :self.opt['hidden_dim']]) + (1 - self.alpha_z) * self.m2_concat(p[:, self.opt['hidden_dim']:]) for p in paths]
      else:
        label_paths_list = [self.m2(p) for p in paths]
      label_paths = torch.stack(label_paths_list, axis=-1)
      return (self.label_atts * label_paths).sum(-1)  # todo maybe generalise this with different attention per class

    elif self.opt['m3_path_dep'] == 'train_centers':
      #get paths of training nodes in feature space
      #for each node work out distance from the C evolving centres
      # make prediction based on these path distances and standard decoder
      label_dist_list = []
      for p in paths:
        base_mask = self.odeblock.odefunc.data.train_mask
        base_av = torch.zeros((self.num_classes, p.shape[-1]), device=self.device)
        # calculate average hidden state per class in the baseline set - [C, d]
        # if base_type == 'train_avg':
        for c in range(self.num_classes):
          base_c_mask = self.odeblock.odefunc.data.y[base_mask] == c
          base_av_c = p[base_mask][base_c_mask].mean(dim=0)
          base_av[c] = base_av_c
        # for every node calcualte the L2 distance - [N, C] and [N, C]
        dist = p.unsqueeze(-1) - base_av.T.unsqueeze(0)
        L2_dist = torch.sqrt(torch.sum(dist ** 2, dim=1))
        label_dist_list.append(L2_dist)

      label_dist_paths = torch.cat(label_dist_list, axis=-1)
      return self.m3(label_dist_paths)

    if self.opt['graph_pool'] == "add":
      z  = global_add_pool(z, self.odeblock.odefunc.data.batch).squeeze(-1)
    elif self.opt['graph_pool'] == "mean":
      z  = global_mean_pool(z, self.odeblock.odefunc.data.batch).squeeze(-1)

    if self.opt['lie_trotter'] == 'gen_2': #if we end in label diffusion block don't need to decode to logits
      if self.opt['lt_gen2_args'][-1]['lt_block_type'] != 'label':
        z = self.m2(z)
    else:
      z = self.m2(z)

    # if self.opt['graph_pool'] == "add":
    #   z  = global_add_pool(z, self.odeblock.odefunc.data.batch).squeeze(-1)
    # elif self.opt['graph_pool'] == "mean":
    #   z  = global_mean_pool(z, self.odeblock.odefunc.data.batch).squeeze(-1)

    #make sure nodewise decoder has non-linearity if not sumation from pooling layers commutes with decoders
    #there is one in GNN_postXN function
    #todo add edwise potential term

    return z