import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
from utils import get_rw_adj
from function_greed_non_linear import ODEFuncGreedNonLin
from greed_reporting_fcts import create_time_lists

class GREEDLTODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(GREEDLTODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)
    self.aug_dim = 2 if opt['augment'] else 1
    ###dummy func just for tracking epoch/wandb_step/get_evol_stats
    # self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)

    self.C = (data.y.max() + 1).item()  #hack!, num class for drift
    funcs_list = []
    # needed for torchdiffeq
    times = []
    steps = []
    # if opt['time2'] is not None:
    if opt['lt_block_times'] is not None:
        self.unpack_blocks(opt)
    # needed for stats and TSNE plots
    self.cum_steps_list, self.cum_time_points, self.cum_time_ticks, self.block_type_list = create_time_lists(self.opt) #self.create_time_lists()
    for block_num, lt2_args in enumerate(self.opt['lt_gen2_args']):
      opt2 = self.opt.as_dict().copy()
      # print(f"lt_block_type {opt2['lt_block_type']}")
      print(f"lt_gen2_args {opt2['lt_gen2_args']}")
      opt2['lt_block_type'] = lt2_args['lt_block_type']
      opt2['time'] = lt2_args['lt_block_time']
      opt2['step_size'] = lt2_args['lt_block_step']
      opt2['hidden_dim'] = lt2_args['lt_block_dimension']
      if 'share_block' in lt2_args.keys():
        opt2['share_block'] = lt2_args['share_block']
      opt2['reports_list'] = lt2_args['reports_list']
      #todo W init
      # opt2['gnl_W_style'] = lt2_args['lt_gnl_W_style']

      odefunc = ODEFuncGreedNonLin
      if opt2['lt_block_type'] == 'label':
        func = odefunc(self.C, self.C, opt2, data, device)
      else:
        func = odefunc(self.aug_dim * opt2['hidden_dim'], self.aug_dim * opt2['hidden_dim'], opt2, data, device)
      if opt2['share_block'] is not None: #todo need to differentiate between W and M2 weights for restart style
        func.load_state_dict(funcs_list[opt2['share_block']].state_dict())

      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                           fill_value=opt2['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)

      func.edge_index = edge_index.to(device) #todo note could inject the rewiring step in here
      func.edge_weight = edge_weight.to(device)
      func.block_num = block_num
      func.cum_steps_list, func.cum_time_points, func.cum_time_ticks, func.block_type_list = self.cum_steps_list, self.cum_time_points, self.cum_time_ticks, self.block_type_list


      funcs_list.append(func)
      time_tensor = torch.tensor([0, opt2['time']], dtype=torch.float).to(device)
      times.append(time_tensor)
      steps.append(lt2_args['lt_block_step'])

    self.funcs = ModuleList(funcs_list)
    self.times = times
    self.steps = steps

    #adding the first func in module list as block attribute to match signature required in run_GNN.py
    # ie model.odeblock.odefunc.epoch
    self.odefunc = self.funcs[0]

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


  def unpack_blocks(self, opt):
    '''function for "double diffusion sweeps'''
    gen2_args = []

    # gen2_args.append({'lt_block_type': 'diffusion', 'lt_block_time': opt['time'], 'lt_block_step': opt['step_size'],
    #  'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': []})
    # if opt['time2'] > 0:
    #   gen2_args.append({'lt_block_type': 'diffusion', 'lt_block_time': opt['time2'], 'lt_block_step': opt['step_size'],
    #                     'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': []})
    # if opt['time3'] > 0:
    #   gen2_args.append({'lt_block_type': 'diffusion', 'lt_block_time': opt['time3'], 'lt_block_step': opt['step_size'],
    #                     'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': [1,2,4,7,8,9,10]})


    for time in opt['lt_block_times']:
      gen2_args.append({'lt_block_type': 'diffusion', 'lt_block_time': time, 'lt_block_step': opt['step_size'],
                        'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': []})

    opt['lt_gen2_args'] = gen2_args



  def set_attributes(self, func, x):
    # if self.opt['function'] in ['greed_linear', 'greed_linear_homo', 'greed_linear_hetero']:
    #   func.set_x_0(x) #this x is actually z
    #   func.set_tau_0()
    #   if self.opt['function'] == 'greed_linear_homo':
    #     func.set_L0()
    #   if self.opt['function'] == 'greed_linear_hetero':
    #     if self.opt['diffusion']:
    #       func.set_L0()
    #       func.Ws = func.set_WS(x)
    #     if self.opt['repulsion']:
    #       func.set_R0()
    #       func.R_Ws = func.set_WS(x)
    if self.opt['function'] in ['greed_non_linear', 'greed_lie_trotter'] and self.opt['gnl_style'] == 'general_graph':
      # if self.opt['gnl_style'] == 'scaled_dot':
      #   self.odeblock.odefunc.Omega = self.odeblock.odefunc.set_scaled_dot_omega()
        func.gnl_W = func.set_gnlWS()
        func.Omega = func.set_gnlOmega()
        func.GNN_postXN = self.odefunc.GNN_postXN
        func.GNN_m2 = self.odefunc.GNN_m2
        # if self.opt['two_hops']: #not currently doing for LT2
        #   self.odeblock.odefunc.gnl_W_tilde = self.odeblock.odefunc.set_gnlWS()
        # if self.opt['gnl_attention']:
        #   self.odeblock.odefunc.set_L0()


  def pass_stats(self, func, block_num):
    func.get_evol_stats = self.odefunc.get_evol_stats
    func.epoch = self.odefunc.epoch
    func.wandb_step = self.odefunc.wandb_step

    end_idx = -1 #this is to not carry over the terminal state as it's replaced by the IC from the net block
    if block_num != 0 and block_num != len(func.opt['lt_gen2_args']): #first block has no preceeding stats.
      prev_func = self.funcs[block_num-1]
      # if func.opt['lt_block_type'] != 'label':
      func.fOmf = prev_func.fOmf[:end_idx]#,:]
      func.attentions = prev_func.attentions[:end_idx]#,:]
      func.L2dist = prev_func.L2dist[:end_idx]#,:]
      func.node_magnitudes = prev_func.node_magnitudes[:end_idx]#,:]
      func.logit_magnitudes = prev_func.logit_magnitudes[:end_idx]#,:]
      func.node_measures = prev_func.node_measures[:end_idx]#,:]

      func.train_accs = prev_func.train_accs[:end_idx]
      func.val_accs = prev_func.val_accs[:end_idx]
      func.test_accs = prev_func.test_accs[:end_idx]
      func.homophils = prev_func.homophils[:end_idx]
      func.entropies = {k: v[:end_idx,:] for k,v in prev_func.entropies.items()}
      func.confusions = [cf[:,:,:end_idx] for cf in prev_func.confusions]

      func.val_dist_mean_feat = prev_func.val_dist_mean_feat[:,:,:end_idx]
      func.val_dist_sd_feat = prev_func.val_dist_sd_feat[:,:,:end_idx]
      func.test_dist_mean_feat = prev_func.test_dist_mean_feat[:,:,:end_idx]
      func.test_dist_sd_feat = prev_func.test_dist_sd_feat[:,:,:end_idx]
      func.val_dist_mean_label = prev_func.val_dist_mean_label[:,:,:end_idx]
      func.val_dist_sd_label = prev_func.val_dist_sd_label[:,:,:end_idx]
      func.test_dist_mean_label = prev_func.test_dist_mean_label[:,:,:end_idx]
      func.test_dist_sd_label = prev_func.test_dist_sd_label[:,:,:end_idx]

      func.paths = prev_func.paths#[:end_idx]


  def forward(self, x):
    integrator = self.train_integrator if self.training else self.test_integrator
    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc

    #loop through lie-trotter blocks
    for block_num, (func, t, step) in enumerate(zip(self.funcs, self.times, self.steps)):
      self.set_x0(func, x)
      self.set_attributes(func, x)
      self.odefunc.reset_gnl_W_eigs(T=0)

      if func.opt['lt_block_type'] == 'label':
        logits, pred = func.predict(x)
        Ek = F.one_hot(pred, num_classes=self.C)
        x = Ek.float()
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

      # non-linearity
      if self.opt['lt_pointwise_nonlin']:
        # x = torch.relu(x)
        x = torch.tanh(x)

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
