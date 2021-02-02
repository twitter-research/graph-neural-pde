"""
Running CGNN model from ICML 20
"""
from ray_tune import main, train_ray_rand
from run_best_ray import run_best_params

for dataset in ['Photo', 'Pubmed', 'Computers', 'CoauthorCS']:
  print('running dataset {}'.format(dataset))
  opt = dict(dataset=dataset)
  opt['name'] = 'CGNN_{}'.format(dataset)
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
  opt['num_samples'] = 1000
  opt['epoch'] = 100
  opt['cpus'] = 2
  opt['gpus'] = 1
  opt['grace_period'] = 10
  opt['reduction_factor'] = 10
  opt['baseline'] = True
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
  # train_ray_rand(opt)
  main(opt)
  # opt = set_search_space(opt)


