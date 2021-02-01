"""
Running CGNN model from ICML 20
"""
from ray_tune import main, train_ray_rand

for dataset in ['Photo', 'Pubmed', 'Computers', 'CoauthorCS', 'ogbn-arxiv']:
  print('running dataset {}'.format(dataset))
  opt = dict(dataset=dataset)
  opt['name'] = 'CGNN_{}'.format(dataset)
  opt['not_lcc'] = False
  # if dataset == 'ogbn-arxiv':
  #   opt['not_lcc'] = False
  # else:
  #   opt['not_lcc'] = True
  opt['num_samples'] = 1
  opt['epoch'] = 1
  opt['num_init'] = 1
  opt['cpus'] = 1
  opt['gpus'] = 0
  opt['grace_period'] = 1
  opt['reduction_factor'] = 10
  opt['baseline'] = True
  opt['metric'] = 'accuracy'
  opt['regularise'] = False
  opt["num_splits"] = 1
  opt['block'] = 'constant'
  opt['function'] = 'laplacian'
  opt['adjoint'] = True
  opt['rewiring'] = None
  opt['self_loop_weight'] = 1
  opt['time'] = 1
  opt['hidden_dim'] = 128
  train_ray_rand(opt)
  # main(opt)
  # opt = set_search_space(opt)



