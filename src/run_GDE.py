"""
Running CGNN model from ICML 20
"""
from ray_tune import main, train_ray_rand
from run_best_ray import run_best_params

# for dataset in ['Photo', 'Pubmed', 'Computers', 'CoauthorCS']:
for dataset in ['Photo', 'Pubmed', 'ogbn-arxiv']:
  print('running dataset {}'.format(dataset))
  opt = dict(dataset=dataset)
  opt['GDE'] = True
  opt['baseline'] = False
  opt['name'] = 'GDE_{}'.format(dataset)
  opt['folder'] = opt['name']
  # opt['not_lcc'] = False
  if dataset == 'ogbn-arxiv':
    opt['not_lcc'] = False
    opt['num_init'] = 2
    opt['num_splits'] = 0
  else:
    opt['not_lcc'] = True
    opt['num_init'] = 1
    opt['num_splits'] = 2
  opt['optimizer'] = 'adam'
  opt['lr'] = 0.04
  opt['decay'] = 0
  opt['num_samples'] = 400
  opt['epoch'] = 100
  opt['cpus'] = 4
  opt['fc_out'] = False
  opt['gpus'] = 1
  opt['grace_period'] = 10
  opt['reduction_factor'] = 10
  opt['metric'] = 'accuracy'
  opt['regularise'] = False
  opt['block'] = 'constant'
  opt['function'] = 'laplacian'
  opt['adjoint'] = True
  opt['rewiring'] = None
  opt['self_loop_weight'] = 1
  opt['time'] = 1
  opt['hidden_dim'] = 128
  opt['alpha'] = 0
  opt['use_labels'] = False
  opt['kinetic_energy'] = None
  opt['jacobian_norm2'] = None
  opt['total_deriv'] = None
  opt['directional_penalty'] = None
  opt['data_norm'] = 'gcn'
  opt['batch_norm'] = False
  opt['method'] = 'rk4'
  opt['step_size'] = 1
  opt['adjoint_step_size'] = 1
  opt['max_iters'] = 100
  opt['max_nfe'] = 300
  opt['no_alpha_sigmoid'] = False
  opt['add_source'] = True
  opt['tol_scale'] = 1
  opt['tol_scale_adjoint'] = 1
  print('running params before tuning of {} '.format(opt))
  main(opt)

