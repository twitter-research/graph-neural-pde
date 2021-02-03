import os
import sys
import argparse
import torch
from GNN_image_pixel import GNN_image_pixel
import time
from torch_geometric.data import DataLoader
from data_image import load_data, load_pixel_data
import pandas as pd
from image_opt_pixel import get_image_opt, opt_perms


def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, dataset, data_test=None):
  model.train()
  loader = DataLoader(dataset, batch_size=model.opt['batch_size'], shuffle=True)

  for batch_idx, batch in enumerate(loader):
    optimizer.zero_grad()
    start_time = time.time()
    if batch_idx > model.opt['train_size'] // model.opt['batch_size']:  # only do for train_size data points
      break

    out = model(batch.x.to(model.device))

    if model.opt['pixel_loss'] in ['binary_sigmoid', '10catlogits', '10catM2','10catkmeans']:
      lf = torch.nn.CrossEntropyLoss()
    elif model.opt['pixel_loss'] == 'MSE':
      lf = torch.nn.MSELoss()

    if model.opt['im_chan'] == 1:
      loss = lf(out[batch.train_mask], batch.y.squeeze()[batch.train_mask].to(model.device))
    if model.opt['im_chan'] == 3:
      loss = 0
      if model.opt['pixel_cat'] == 2 and model.opt['pixel_loss'] == 'binary_sigmoid':
        loss += lf(torch.stack((out[:,0],out[:,3]),dim=1)[batch.train_mask], batch.y[:,0].squeeze()[batch.train_mask].to(model.device))
        loss += lf(torch.stack((out[:,1],out[:,4]),dim=1)[batch.train_mask], batch.y[:,1].squeeze()[batch.train_mask].to(model.device))
        loss += lf(torch.stack((out[:,2],out[:,5]),dim=1)[batch.train_mask], batch.y[:,2].squeeze()[batch.train_mask].to(model.device))

      elif model.opt['pixel_cat'] == 10 and model.opt['pixel_loss'] == '10catlogits':
        for i in range(10):
          pass #old method taking pixel slices was wrong anyway
          # loss += lf(torch.stack((out[:, i], out[:, i+9]), dim=1)[batch.train_mask].repeat(3),batch.y.view(-1,1)[batch.train_mask].repeat(3))
          # A = out[batch.train_mask.repeat(3)]
          # B = batch.y.view(-1, 1).squeeze()[batch.train_mask.repeat(3)].to(model.device)
          # loss += lf(A,B)
          # loss += lf(torch.stack((out[:, i], out[:, i + 9]), dim=1)[batch.train_mask.repeat(3)],
          #      batch.y.view(-1, 1).squeeze()[batch.train_mask.repeat(3)])

      elif model.opt['pixel_cat'] > 1 and model.opt['pixel_loss'] == '10catM2':
        A = out[batch.train_mask]
        B = batch.y.view(-1, 1).squeeze()[batch.train_mask].to(model.device)
        loss += lf(A, B)

      elif model.opt['pixel_loss'] == '10catkmeans':
        z = out[batch.train_mask].unsqueeze(1)
        batch_centers = batch.label_centers
        centers_idx = batch.batch[batch.train_mask]
        each_batch_centers = []
        pixel_cat = model.opt['pixel_cat']
        for i in range(pixel_cat):
          batch_center = batch_centers[centers_idx * pixel_cat + i].unsqueeze(1)
          each_batch_centers.append(batch_center)
        all_batch_centers = torch.cat(each_batch_centers,dim=1)
        logits = torch.sum((z-all_batch_centers)**2,dim=2)
        logits = 1 / (logits ** 2 + 1e-5)
        loss += lf(logits, batch.y[batch.train_mask].squeeze())

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    # if model.opt['testing_code']:
    if batch_idx % 1 == 0:
      train_acc = pixel_test(model, batch, "batch","train")
      test_acc = pixel_test(model, batch, "batch","test")
      log = 'Batch Index: {}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Test: {:.4f}'
      print(log.format(batch_idx, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, test_acc))
    #   elif batch_idx % 1 == 0:
    #     print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum,
    #                                                                           time.time() - start_time))
    # else:
    #   # if batch_idx % (model.opt['train_size'] / model.opt['batch_size'] / 10) == 0:
    #   # # if batch_idx % 5 == 0:
    #   #   test_acc = test(model, data_test)
    #   #   log = 'Batch Index: {}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Test: {:.4f}'
    #   #   print(log.format(batch_idx, time.time() - start_time, loss, model.fm.sum, model.bm.sum, test_acc))
    #   if batch_idx % 1 == 0:
    #     print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum,
    #                                                                           time.time() - start_time))
    # print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum,
    #                                                                       time.time() - start_time))

  return loss.item()


@torch.no_grad()
def test(model, dataset):
  data = dataset.data
  test_size = data.y.shape[0]
  batch_size = model.opt['batch_size']
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  total_correct = 0
  for batch_idx, batch in enumerate(loader):
    if batch_idx > model.opt['test_size'] // model.opt['batch_size']:  # only do this for 1st batch/epoch
      break
    model.eval()
    logits, accs = model(batch.x.to(model.device)), []
    pred = logits.max(1)[1]
    total_correct += pred.eq(batch.y.T.to(model.device)).sum().item()
  accs = total_correct / test_size
  return accs


@torch.no_grad()
def pixel_test(model, data, batchorTest, trainorTest="test"):
  total_correct = 0

  if model.opt['im_chan'] == 1:
    mask = data.test_mask if trainorTest == "test" else data.train_mask
    test_size = data.y[mask].shape[0]
    model.eval()
    logits = model(data.x.to(model.device))[mask]
    pred = logits.max(1)[1]
    total_correct += pred.eq(data.y[mask].T.to(model.device)).sum().item()

  elif model.opt['im_chan'] == 3 and model.opt['pixel_loss'] in ['10catM2','10catlogits']:

    mask = data.test_mask if trainorTest == "test" else data.train_mask

    test_size = data.y[mask].shape[0]
    model.eval()
    logits = model(data.x.to(model.device))[mask]
    pred = logits.max(1)[1]
    total_correct += pred.eq(data.y[mask].T.to(model.device)).sum().item()

  elif model.opt['im_chan'] == 3 and model.opt['pixel_loss'] == '10catkmeans':
    model.eval()
    mask = data.test_mask if trainorTest == "test" else data.train_mask
    out = model(data.x.to(model.device))
    z = out[mask].unsqueeze(1)
    batch_centers = data.label_centers
    test_size = data.y[mask].shape[0]
    if batchorTest == "batch":
        centers_index = data.batch[mask]
    elif batchorTest == "test":
        len_data = len(data.target)
        pixels = model.opt['im_height'] * model.opt['im_width']
        centers_index = torch.LongTensor(torch.arange(len_data*pixels)//pixels)[mask]

    each_batch_centers = []
    pixel_cat = model.opt['pixel_cat']
    for i in range(pixel_cat):
      batch_center = batch_centers[centers_index * pixel_cat + i].unsqueeze(1)
      each_batch_centers.append(batch_center)
    all_batch_centers = torch.cat(each_batch_centers, dim=1)
    logits = torch.sum((z - all_batch_centers) ** 2, dim=2)
    logits = 1 / (logits ** 2 + 1e-5)

    pred = logits.max(1)[1]
    total_correct += pred.eq(data.y[mask].T.to(model.device)).sum().item()

    #old method taking slices
    # test_size = data.y.shape[0] * data.y.shape[1]
    # model.eval()
    # out = model(data.x.to(model.device))
    # if model.opt['pixel_cat'] == 2 and model.opt['pixel_loss'] == 'binary_sigmoid':
    #   logitsR = torch.stack((out[:, 0], out[:, 3]), dim=1)
    #   logitsG = torch.stack((out[:, 1], out[:, 4]), dim=1)
    #   logitsB = torch.stack((out[:, 2], out[:, 5]), dim=1)
    #   predR = logitsR.max(1)[1]
    #   predG = logitsG.max(1)[1]
    #   predB = logitsB.max(1)[1]
    #   total_correct += predR.eq(data.y[:,0].T.to(model.device)).sum().item()
    #   total_correct += predG.eq(data.y[:,1].T.to(model.device)).sum().item()
    #   total_correct += predB.eq(data.y[:,2].T.to(model.device)).sum().item()
    # elif model.opt['pixel_cat'] == 10 and model.opt['pixel_loss'] == '10catlogits':
    #   if batchorTest == "batch":
    #     span = model.opt['im_height'] * model.opt['im_width'] * model.opt['batch_size']
    #   elif batchorTest == "test":
    #     span = model.opt['im_height'] * model.opt['im_width'] * model.opt['train_size']
    #
    #   logitsR = out[0:span, :]
    #   logitsG = out[span:2*span, :]
    #   logitsB = out[2*span:3*span, :]
    #   predR = logitsR.max(1)[1]
    #   predG = logitsG.max(1)[1]
    #   predB = logitsB.max(1)[1]
    #   total_correct += predR.eq(data.y[:, 0].T.to(model.device)).sum().item()
    #   total_correct += predG.eq(data.y[:, 1].T.to(model.device)).sum().item()
    #   total_correct += predB.eq(data.y[:, 2].T.to(model.device)).sum().item()

  accs = total_correct / test_size
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


def main(opt):
  csv_path = '../pixels/models.csv'
  if os.path.exists(csv_path):
    try:
      os.rename(csv_path, '../pixels/temp_models.csv')
      os.rename('../pixels/temp_models.csv', csv_path)
    except OSError:
      print(f'Error {csv_path} is still open, please close and rerun.')
      sys.exit(1)

  opt_permutations = opt_perms(opt)
  for opt_perm, opt in opt_permutations.items():
    print(f"This is run permutation {opt['im_dataset']} {opt['block']} {opt['function']}")
    opt = get_image_opt(opt)

    print("Loading Data")
    pixel_data = load_pixel_data(opt)
    # data_train, data_test = load_data(opt)
    print("creating GNN model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader = DataLoader(data_train, batch_size=opt['batch_size'], shuffle=True)
    loader = DataLoader(pixel_data, batch_size=opt['batch_size'], shuffle=True)

    for batch_idx, batch in enumerate(loader):
      break
    batch.to(device)
    edge_index_gpu = batch.edge_index
    edge_attr_gpu = batch.edge_attr
    if edge_index_gpu is not None: edge_index_gpu.to(device)
    if edge_attr_gpu is not None: edge_index_gpu.to(device)
    model = GNN_image_pixel(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                      batch.edge_attr, device).to(device)

    print(opt)
    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])

    timestr = time.strftime("%Y%m%d_%H%M%S")
    # save model - params only - no point repeatedly saving data
    data_name = opt['im_dataset']
    blck = opt['block']
    fct = opt['function']
    model_key = f"{timestr}"
    savefolder = f"../pixels/{timestr}_{data_name}_{blck}_{fct}"
    savepath = f"{savefolder}/model_{model_key}"
    try:
      os.mkdir(savefolder)
    except OSError:
      print("Creation of the directory %s failed" % savefolder)
    else:
      print("Successfully created the directory %s " % savefolder)

    for epoch in range(opt['epoch']):
      print("Epoch {}".format(epoch))
      start_time = time.time()
      loss = train(model, optimizer, pixel_data)
      test_acc = pixel_test(model, pixel_data.data, batchorTest="test", trainorTest="test")

      log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Test: {:.4f}'
      print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, test_acc))

      if epoch % 8 == 0:
        torch.save(model.state_dict(), f"{savepath}_epoch{epoch}.pt")

    # save run details to csv
    opt['model_key'] = model_key
    opt['Test Acc'] = test_acc
    df = pd.DataFrame({k: [v] for k, v in opt.items()})
    cols = list(df)
    top_cols = ['model_key', 'testing_code', 'im_dataset', 'function', 'block', 'simple', 'batched',
                'diags', 'batch_size', 'train_size', 'test_size', 'Test Acc', 'alpha']
    for head in reversed(top_cols):
      cols.insert(0, cols.pop(cols.index(head)))
    df = df.loc[:, cols]
    header = False if os.path.exists(csv_path) else True
    df.to_csv(csv_path, mode='a', header=header)

    # print("creating GNN model")
    # batch.to(device)
    # edge_index_gpu = batch.edge_index
    # edge_attr_gpu = batch.edge_attr
    # if edge_index_gpu is not None: edge_index_gpu.to(device)
    # if edge_attr_gpu is not None: edge_index_gpu.to(device)
    #
    # model = GNN_image(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
    #                   batch.edge_attr, device).to(device)  #
    # model.load_state_dict(torch.load(savepath))
    # out = model(batch.x)
    # model.eval()
    # print_image_T(model, Graph_test, opt, height=2, width=3,)
    # animation = print_image_path(model, Graph_test, height=2, width=3, frames=10)
    # animation.save(f'../images/Graph{exdataset}_ani_{timestr}.gif', writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=0.5)
    # plot_att_heat(model)
    print(f"Test acc {test_acc}")
  return test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument('--use_cora_defaults', action='store_true',
  #                     help='Whether to run with best params for cora. Overrides the choice of dataset')
  # parser.add_argument('--dataset', type=str, default='Cora',
  #                     help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--alpha_sigmoid', type=bool, default=True, help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun",
    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--simple', type=bool, default=True,
                      help='If try get rid of alpha param and the beta*x0 source term')
  parser.add_argument('--step_size', type=float, default=1, help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')

  # SDE args
  parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
  parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
  parser.add_argument('--adaptive', type=bool, default=False, help='use adaptive step sizes')
  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', type=bool, default=False,
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument("--max_nfe", type=int, default=5000, help="Maximum number of function evaluations allowed.")
  parser.add_argument('--reweight_attention', type=bool, default=False,
                      help="multiply attention scores by edge weights before softmax")
  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")
  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")
  # rewiring args
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  # visualisation args
  parser.add_argument('--testing_code', type=bool, default=False, help='run on limited size training/test sets')
  parser.add_argument('--use_image_defaults', default='True', help='sets as per function get_image_opt')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
  parser.add_argument('--train_size', type=int, default=128, help='Batch size')
  parser.add_argument('--test_size', type=int, default=128, help='Batch size')
  parser.add_argument('--batched', type=bool, default=True, help='Batching')
  parser.add_argument('--im_width', type=int, default=28, help='im_width')
  parser.add_argument('--im_height', type=int, default=28, help='im_height')
  parser.add_argument('--im_chan', type=int, default=1, help='im_height')
  parser.add_argument('--num_class', type=int, default=10, help='im_height')
  parser.add_argument('--diags', type=bool, default=False, help='Edge index include diagonal diffusion')
  parser.add_argument('--im_dataset', type=str, default='MNIST', help='MNIST, CIFAR')
  parser.add_argument('--num_nodes', type=int, default=28 ** 2, help='im_width')

  args = parser.parse_args()
  opt = vars(args)
  main(opt)
