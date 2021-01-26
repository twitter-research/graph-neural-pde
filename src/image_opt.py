
def opt_perms(opt):
  im_dataset = ['MNIST','CIFAR']
  blocks = ['transformer', 'laplacian']
  functions = ['constant', 'attention']


  opt_perms = {}
  for dataset in im_dataset:
    opt['im_dataset'] = dataset

    opt['block'] = 'transformer'
    opt['function'] = 'attention'
    opt_perms[f"{dataset}_{opt['block']}_{opt['function']}"] = opt

    opt['block'] = 'transfomer'
    opt['function'] = 'constant'
    opt_perms[f"{dataset}_{opt['block']}_{opt['function']}"] = opt

    opt['block'] = 'laplacian'
    opt['function'] = 'constant'
    opt_perms[f"{dataset}_{opt['block']}_{opt['function']}"] = opt

    #
    # for block in blocks:
    #   for function in functions:
    #     if opt['block'] != and opt['function'] != :
    #       opt['im_dataset'] = dataset
    #       opt['block'] = block
    #       opt['function'] = function
    # opt_perms[dataset+"_"+block+"_"+function] = opt
  return opt_perms


def get_image_opt(opt):
  opt['testing_code'] = False  # True #to work with smaller dataset

  opt['simple'] = True #True
  opt['adjoint'] = True

  opt['method'] = 'rk4'
  opt['adjoint_method'] = 'rk4'
  opt['adjoint'] = True
  opt['step_size'] = 1.0
  opt['adjoint_step_size'] = 1.0

  # opt['tol_scale'] = 2.0 #., help = 'multiplier for atol and rtol')
  # opt['tol_scale_adjoint'] = 2.0

  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  # opt['alpha'] = 0
  opt['time'] = 5 #2
  opt['augment'] = False #True   #False need to view image
  opt['attention_dropout'] = 0

  opt['epoch'] = 2 #2 #2 #3 #1
  opt['batched'] = True
  if opt['testing_code']:
    opt['batch_size'] = 64  # 64 #64  # doing batch size for mnist
    opt['train_size'] = 512#0 #128 #10240 #512 #10240
    opt['test_size'] = 128#0  #512#64#128

  assert (opt['train_size']) % opt['batch_size'] == 0, "train_size needs to be multiple of batch_size"
  assert (opt['test_size']) % opt['batch_size'] == 0, "test_size needs to be multiple of batch_size"

  if opt['im_dataset'] == 'MNIST':
    # http://yann.lecun.com/exdb/mnist/
    opt['im_width'] = 28
    opt['im_height'] = 28
    opt['im_chan'] = 1
    opt['hidden_dim'] = 1
    opt['num_feature'] = 1
    opt['num_class'] = 10
    if not opt['testing_code']:
      opt['batch_size'] = 100
      opt['train_size'] = 60000
      opt['test_size'] = 10000

  elif opt['im_dataset'] == 'CIFAR':
    # https://www.cs.toronto.edu/~kriz/cifar.html
    opt['im_width'] = 32
    opt['im_height'] = 32
    opt['im_chan'] = 3
    opt['hidden_dim'] = 3
    opt['num_feature'] = 3
    opt['num_class'] = 10
    if not opt['testing_code']:
      opt['batch_size'] = 100
      opt['train_size'] = 50000
      opt['test_size'] = 10000

  opt['num_nodes'] = opt['im_height'] * opt['im_width'] # * opt['im_chan']
  opt['diags'] = True

  return opt