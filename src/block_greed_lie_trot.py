import torch
from torch.nn import ModuleList
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
from utils import get_rw_adj
# from function_greed_non_linear import ODEFuncGreedNonLin
from function_greed_non_linear_lie_trotter import ODEFuncGreedLieTrot

class GREEDLTODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(GREEDLTODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)
    self.aug_dim = 2 if opt['augment'] else 1

    self.C = (data.y.max() + 1).item()  #hack!, num class for drift
    funcs = []
    times = []
    steps = []
    for lt2_args in self.opt['lt_gen2_args']:
      opt2 = self.opt.as_dict().copy()
      opt2['lt_block_type'] = lt2_args['lt_block_type']
      opt2['time'] = lt2_args['lt_block_time']
      opt2['step_size'] = lt2_args['lt_block_step']
      opt2['hidden_dim'] = lt2_args['lt_block_dimension']
      odefunc = ODEFuncGreedLieTrot
      if opt2['lt_block_type'] == 'label':
        func = odefunc( self.C, self.C, opt2, data, device)
      else:
        func = odefunc(self.aug_dim * opt2['hidden_dim'], self.aug_dim * opt2['hidden_dim'], opt2, data, device)
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                           fill_value=opt2['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
      func.edge_index = edge_index.to(device)
      func.edge_weight = edge_weight.to(device)
      funcs.append(func)
      time_tensor = torch.tensor([0, opt2['time']], dtype=torch.float).to(device)
      times.append(time_tensor)
      steps.append(lt2_args['lt_block_step'])

    self.funcs = ModuleList(funcs)
    self.current_block_num = 0
    self.times = times
    self.steps = steps
    #adding the first func in module list as block attribute to match signature required in run_GNN.py
    self.odefunc = self.funcs[0]
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()

  def set_x0(self, odefunc, x0):
    odefunc.x0 = x0.clone().detach()
    # reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_attributes(self, func, x):
    if self.opt['function'] in ['greed_linear', 'greed_linear_homo', 'greed_linear_hetero']:
      func.set_x_0(x) #this x is actually z
      func.set_tau_0()
      if self.opt['function'] == 'greed_linear_homo':
        func.set_L0()
      if self.opt['function'] == 'greed_linear_hetero':
        if self.opt['diffusion']:
          func.set_L0()
          func.Ws = func.set_WS(x)
        if self.opt['repulsion']:
          func.set_R0()
          func.R_Ws = func.set_WS(x)
    if self.opt['function'] in ['greed_non_linear', 'greed_lie_trotter'] and self.opt['gnl_style'] == 'general_graph':
          func.gnl_W = func.set_gnlWS()
          func.GNN_m2 = self.odefunc.GNN_m2

  def pass_stats(self, func, block_num):
    func.get_evol_stats = self.odefunc.get_evol_stats
    func.epoch = self.odefunc.epoch
    func.wandb_step = self.odefunc.wandb_step

    if block_num != 0:
      prev_func = self.funcs[block_num-1]
      if func.opt['lt_block_type'] != 'label':
        func.fOmf = prev_func.fOmf
        func.attentions = prev_func.attentions
        func.L2dist = prev_func.L2dist
        func.node_magnitudes = prev_func.node_magnitudes
        func.node_measures = prev_func.node_measures

      func.train_accs = prev_func.train_accs
      func.val_accs = prev_func.val_accs
      func.test_accs = prev_func.test_accs
      func.homophils = prev_func.homophils
      func.entropies = prev_func.entropies
      func.confusions = prev_func.confusions

      func.val_dist_mean_feat = prev_func.val_dist_mean_feat
      func.val_dist_sd_feat = prev_func.val_dist_sd_feat
      func.test_dist_mean_feat = prev_func.test_dist_mean_feat
      func.test_dist_sd_feat = prev_func.test_dist_sd_feat
      func.val_dist_mean_label = prev_func.val_dist_mean_label
      func.val_dist_sd_label = prev_func.val_dist_sd_label
      func.test_dist_mean_label = prev_func.test_dist_mean_label
      func.test_dist_sd_label = prev_func.test_dist_sd_label

  def forward(self, x):
    integrator = self.train_integrator if self.training else self.test_integrator
    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc

    #loop through lie-trotter blocks
    for block_num, (func, t, step) in enumerate(zip(self.funcs, self.times, self.steps)):
      self.set_x0(func, x)
      self.set_attributes(func, x)
      if func.opt['lt_block_type'] == 'label':
        x, _ = func.predict(x)
        self.set_x0(func, x) #reset given now in label space
        self.set_attributes(func, x)

      if self.opt['wandb_track_grad_flow'] and self.odefunc.epoch in self.opt['wandb_epoch_list'] and self.odefunc.get_evol_stats:
        with torch.no_grad():
          self.pass_stats(func, block_num)

      reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))
      state = (x,) + reg_states if self.training and self.nreg > 0 else x

      if self.opt["adjoint"] and self.training:
        state_dt = integrator(
          func, state, t,
          method=self.opt['method'],
          options={'step_size': step},
          adjoint_method=self.opt['adjoint_method'],
          adjoint_options={'step_size': self.opt['adjoint_step_size']},
          atol=self.atol,
          rtol=self.rtol,
          adjoint_atol=self.atol_adjoint,
          adjoint_rtol=self.rtol_adjoint)
      else:
        state_dt = integrator(
          func, state, t,
          method=self.opt['method'],
          options={'step_size': step},
          atol=self.atol,
          rtol=self.rtol)
      #terminal state as initial condition for next block
      if self.training and self.nreg > 0:
        x = state_dt[0][1]
        reg_states = tuple(st[1] for st in state_dt[1:])
      else:
        x = state_dt[1]

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple(st[1] for st in state_dt[1:])
      return z, reg_states
    else:
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
