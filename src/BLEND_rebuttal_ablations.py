import argparse
import numpy as np
import torch
import torch.nn.functional as F
import time
import json
import pandas as pd
from GNN import GNN
from GNN_early import GNNEarly
from data import get_dataset, set_train_val_test_split
from ogb.nodeproppred import Evaluator
from graph_rewiring import apply_gdc, apply_beltrami
from best_params import  best_params_dict
from run_GNN import print_model_params, get_optimizer, test, test_OGB, train
from statistics import mean, stdev

def main(opt):

  dataset = get_dataset(opt, '../data', opt['not_lcc'])
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)

  if not opt['planetoid_split'] and opt['dataset'] in ['Cora','Citeseer','Pubmed']:
    dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  data = dataset.data.to(device)

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  patience_counter = 0
  patience = 100
  for epoch in range(1, opt['epoch']):
    if patience_counter == patience:
        break
    start_time = time.time()
    this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
    loss = train(model, optimizer, data, pos_encoding)

    if opt["no_early"]:
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)
      best_time = opt['time']
      if tmp_val_acc > val_acc:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        test_acc = tmp_test_acc
      else:
        patience_counter += 1

    else:
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)
      if tmp_val_acc > val_acc:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        test_acc = tmp_test_acc
        best_time = opt['time']
      else:
        patience_counter += 1

      if model.odeblock.test_integrator.solver.best_val > val_acc:
        best_epoch = epoch
        val_acc = model.odeblock.test_integrator.solver.best_val
        test_acc = model.odeblock.test_integrator.solver.best_test
        train_acc = model.odeblock.test_integrator.solver.best_train
        best_time = model.odeblock.test_integrator.solver.best_time

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(val_acc, test_acc, best_epoch))
  return train_acc, val_acc, test_acc

#ODE solver ablation
def ODE_solver_ablation(cmd_opt):
    datas = ['Cora','Citeseer','Pubmed','CoauthorCS','Computers','Photo']
    # datas = ['Pubmed','CoauthorCS','Computers','Photo']
    methods = ['best', 'euler']

    rows = []
    for i, ds in enumerate(datas):
        best_opt = best_params_dict[ds]
        opt = {**cmd_opt, **best_opt}
        opt['no_early'] = True #no implementation of early stop solver for explicit euler

        best_method = opt['method']
        best_step = opt['step_size']
        best_adj_method = opt['adjoint_method']
        best_step_adj = opt['adjoint_step_size']

        for method in methods:
            if method == 'best':
                opt['method'] = best_method
                opt['step_size'] = best_step
                opt['adjoint_method'] = best_adj_method
                opt['adjoint_step_size'] = best_step_adj
            elif method == 'euler':
                opt['method'] = 'euler'
                opt['step_size'] = 1.0
                opt['adjoint_method'] = 'euler'
                opt['adjoint_step_size'] = 1.0

            for it in range(2):
                opt['epoch'] = 5
                print(f"Running Best Params for {ds}")
                train_acc, val_acc, test_acc = main(opt)
                row = [ds, it, opt['method'], opt['step_size'], opt['adjoint_method'], opt['adjoint_step_size'], train_acc, val_acc, test_acc]
                rows.append(row)

        df = pd.DataFrame(rows, columns = ['dataset','iteration','method','step_size','adjoint_method','adjoint_step_size', 'train_acc', 'val_acc', 'test_acc'])
        pd.set_option('display.max_columns', None)
        print(df)
        df.to_csv(f"../ablations/ODE_solver_data_{ds}.csv")

        mean_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'], index=["dataset","method"],
                               aggfunc={'train_acc': np.mean, 'val_acc': np.mean, 'test_acc': np.mean}, margins=True)
        print(mean_table)
        mean_table.to_csv(f"../ablations/ODE_solver_mean_{ds}.csv")

        std_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'], index=["dataset","method"],
                               aggfunc={'train_acc': np.std,'val_acc': np.std,'test_acc': np.std}, margins=True)
        print(std_table)
        std_table.to_csv(f"../ablations/ODE_solver_std_{ds}.csv")

#attention type ablation
def attention_ablation(cmd_opt):
    datas = ['Cora']#,'Citeseer','Pubmed','CoauthorCS','Computers','Photo']
    attentions = ['scaled_dot','cosine_sim','pearson', 'exp_kernel']

    rows = []
    for i, ds in enumerate(datas):
        best_opt = best_params_dict[ds]
        opt = {**cmd_opt, **best_opt}

        for attention in attentions:
            opt['attention_type'] = attention
            for it in range(8):
                print(f"Running Best Params for {ds}")
                train_acc, val_acc, test_acc = main(opt)
                row = [ds, it, opt['attention_type'], train_acc, val_acc, test_acc]
                rows.append(row)

        df = pd.DataFrame(rows, columns = ['dataset','iteration','attention_type','train_acc', 'val_acc', 'test_acc'])
        pd.set_option('display.max_columns', None)
        print(df)
        df.to_csv(f"../ablations/attention_data_{ds}.csv")

        mean_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'], index=["dataset","attention_type"],
                               aggfunc={'train_acc': np.mean, 'val_acc': np.mean, 'test_acc': np.mean}, margins=True)
        print(mean_table)
        mean_table.to_csv(f"../ablations/attention_mean_{ds}.csv")

        std_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'], index=["dataset","attention_type"],
                               aggfunc={'train_acc': np.std,'val_acc': np.std,'test_acc': np.std}, margins=True)
        print(std_table)
        std_table.to_csv(f"../ablations/attention_std_{ds}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    # GNN args
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention, SDE')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
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
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")

    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                        help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")

    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

    args = parser.parse_args()

    opt = vars(args)

    # ODE_solver_ablation(opt)
    attention_ablation(opt)
