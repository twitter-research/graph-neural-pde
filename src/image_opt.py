
def get_image_opt(opt):

  opt['im_dataset'] =  'MNIST' #'MNIST'  #datasets = ['MNIST','CIFAR']
  opt['testing_code'] = True #False #True #to work with smaller dataset
  opt['function'] = 'transformer' #'laplacian' #'transformer'
  opt['block'] = 'attention' #'constant' #'attention' 'mixed
  opt['simple'] = True

  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  opt['alpha'] = 0.918
  opt['time'] = 1
  opt['augment'] = False #True   #False need to view image
  opt['attention_dropout'] = 0
  opt['adjoint'] = False

  opt['epoch'] = 2 #3 #1
  opt['batched'] = True
  if opt['testing_code']:
    opt['batch_size'] = 64  # 64 #64  # doing batch size for mnist
    opt['train_size'] = 128 #10240 #512 #10240
    opt['test_size'] = 128  #512#64#128

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
