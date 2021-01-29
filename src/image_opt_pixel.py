
def opt_perms(opt):
  """
  im_dataset = ['MNIST','CIFAR']
  blocks = ['attention', 'constant']
  functions = ['laplacian', 'transformer']
  """
  opt_perms = {}
  opt1 = opt.copy()
  opt1['im_dataset'] = 'MNIST'
  opt1['block'] = 'constant'
  opt1['function'] = 'laplacian'
  opt_perms[f"{opt1['im_dataset']}_{opt1['block']}_{opt1['function']}"] = opt1
  opt2 = opt.copy()
  opt2['im_dataset'] = 'MNIST'
  opt2['block'] = 'attention'
  opt2['function'] = 'laplacian'
  opt_perms[f"{opt2['im_dataset']}_{opt2['block']}_{opt2['function']}"] = opt2
  opt3 = opt.copy()
  opt3['im_dataset'] = 'MNIST'
  opt3['block'] = 'attention'
  opt3['function'] = 'transformer'
  opt_perms[f"{opt3['im_dataset']}_{opt3['block']}_{opt3['function']}"] = opt3
  opt4 = opt.copy()
  opt4['im_dataset'] = 'CIFAR'
  opt4['block'] = 'constant'
  opt4['function'] = 'laplacian'
  opt_perms[f"{opt4['im_dataset']}_{opt4['block']}_{opt4['function']}"] = opt4
  opt5 = opt.copy()
  opt5['im_dataset'] = 'CIFAR'
  opt5['block'] = 'attention'
  opt5['function'] = 'laplacian'
  opt_perms[f"{opt5['im_dataset']}_{opt5['block']}_{opt5['function']}"] = opt5
  opt6 = opt.copy()
  opt6['im_dataset'] = 'CIFAR'
  opt6['block'] = 'attention'
  opt6['function'] = 'transformer'
  opt_perms[f"{opt6['im_dataset']}_{opt6['block']}_{opt6['function']}"] = opt6

    # for dataset in im_dataset:
    # for block in blocks:
    #   for function in functions:
    #     if opt['block'] != and opt['function'] != :
    #       opt['im_dataset'] = dataset
    #       opt['block'] = block
    #       opt['function'] = function
    # opt_perms[dataset+"_"+block+"_"+function] = opt
  return opt_perms

def get_image_opt(opt):
  opt['testing_code'] = True #True  # True #to work with smaller dataset

  opt['pixel_cat'] = 10 #10 #2 #[2 for binary_sigmoid, 10 for'10catM2','10catlogits' ]
  opt['pixel_loss'] = '10catlogits'#'binary_sigmoid' #'10catlogits' #  ['binary_sigmoid','10catM2','10catlogits','MSE']

  opt['simple'] = True #True
  # opt['alpha'] = 0   ###note alpha and beta manually turned off at the function level
  opt['adjoint'] = True

  opt['method'] = 'rk4'
  opt['adjoint_method'] = 'rk4'
  opt['adjoint'] = True
  opt['step_size'] = 1.0
  opt['adjoint_step_size'] = 1.0
  opt['max_iters'] = 5000
  # opt['tol_scale'] = 2.0 #., help = 'multiplier for atol and rtol')
  # opt['tol_scale_adjoint'] = 2.0

  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.0 #0.555
  opt['time'] = 16 #5 #2
  opt['augment'] = False #True   #False need to view image
  opt['attention_dropout'] = 0

  opt['epoch'] = 32 #2 #2 #3 #1
  opt['batched'] = True
  if opt['testing_code']:
    opt['batch_size'] = 64 #64  # doing batch size for mnist
    opt['train_size'] = 1024 #512 #0 #128 #10240 #512 #10240
    opt['test_size'] = 128 #0  #512#64#128

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