import argparse
import datetime, time
import torch
import torch_sparse
import numpy as np
import pandas as pd
from greed_params import greed_run_params, not_sweep_args, tf_ablation_args, default_params
from torch_geometric.utils import homophily, add_remaining_self_loops, to_undirected, to_dense_adj
import wandb

from run_GNN import main, unpack_gcn_params, unpack_greed_params
from data import get_dataset
from heterophilic import get_fixed_splits
from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from GNN_GCN import GCN, MLP, GAT
from GNN_GCNMLP import GNNMLP

def run_track_flow_experiments():
    opt = default_params()
    #wandb args
    opt['wandb'] = True #False #True
    opt['wandb_track_grad_flow'] = True
    opt['wandb_watch_grad'] = False
    opt['run_track_reports'] = True
    opt['wandb_reports'] = True
    opt['save_local_reports'] = False
    opt['wandb_epoch_list'] = [1,2,4,8,16,32,64,128]
    opt['wandb_entity'] = "graph_neural_diffusion"
    opt['wandb_project'] = "reporting_runs"
    opt['wandb_group'] = "reporting_group"
    DT = datetime.datetime.now()
    opt['wandb_run_name'] = DT.strftime("%m%d_%H%M%S_") + "wandb_best_BLEND_params"  # "wandb_log_gradflow_test3"

    #experiments args
    opt['use_best_params'] = False
    opt['method'] = 'euler'
    opt['step_size'] = 1.0
    opt['epoch'] = 129
    opt['num_splits'] = 1
    opt['optimizer'] = 'adam'
    opt['lr'] = 0.005
    opt['dropout'] = 0.6
    opt['decay'] = 0.0

    #GNN args
    opt['block'] = 'constant'
    opt['function'] = 'greed_non_linear'
    opt['gnl_style'] = 'general_graph'
    opt['add_source'] = True
    opt['use_mlp'] = False
    opt['XN_no_activation'] = True
    opt['m2_mlp'] = False
    opt['self_loop_weight'] = 0.0
    opt['no_early'] = True
    opt['gnl_thresholding'] = False

    for data in ['chameleon', 'Cora']:
        if data == 'chameleon':
            opt['geom_gcn_splits'] = True
        elif data == 'Cora':
            opt['geom_gcn_splits'] = False
        opt['dataset'] = data

        for gnl_measure in ['ones', 'deg_poly', 'nodewise', 'nodewise_exp', 'deg_poly_exp']:
            opt['gnl_measure'] = gnl_measure
            for drift in [True]: #False, True]:
                opt['drift'] = drift
                for gnl_W_style in ['sum', 'prod', 'k_diag', 'k_block', 'cgnn']:#'sum', 'prod', 'k_diag', 'k_block' 'cgnn']:
                    opt['gnl_W_style'] = gnl_W_style
                    if opt['gnl_W_style'] == 'k_block':
                        opt['k_blocks'] = 5
                        opt['block_size'] = 5
                    elif opt['gnl_W_style'] == 'k_diag':
                        opt['k_diags'] = 21
                    for time in [3, 8]:
                        opt['time'] = time
                        for hidden_dim in [64, 512]:
                            opt['hidden_dim'] = hidden_dim
                            opt = tf_ablation_args(opt)
                            opt['gnl_savefolder'] = f"{data}_{gnl_measure}_drift{str(drift)}_W{gnl_W_style}_t{str(time)}_hd{str(hidden_dim)}"
                            main(opt)

def wall_clock_ablation(opt):
    if 'wandb_run_name' in opt.keys():
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
    else:
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               reinit=True, config=opt, allow_val_change=True)  # required when update config

    opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!
    if opt['gcn_params']: #temp function for GCN ablation
        unpack_gcn_params(opt)
    if opt['greed_params']: #temp function for GCN ablation
        unpack_greed_params(opt)

    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    if opt['dataset'] in ['chameleon','squirrel','other hetero?']:
        ### added self loops and make undirected for chameleon
        dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    if opt['geom_gcn_splits']:
        if opt['dataset'] == "Citeseer":
            dataset = get_dataset(opt, '../data', opt['not_lcc']) #geom-gcn citeseer uses splits over LCC and not_LCC so need to repload each split
        data = get_fixed_splits(dataset.data, opt['dataset'], 0)
        dataset.data = data

    data = dataset.data
    feat = data.x
    pos_encoding = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt['function'] in ['gcn']:
        model = GCN(opt, dataset, hidden=[opt['hidden_dim']], dropout=opt['dropout'], device=device).to(device)
    elif opt['function'] in ['mlp']:
        model = MLP(opt, dataset, device=device).to(device)
    elif opt['function'] in ['gcn2', 'gcn_dgl', 'gcn_res_dgl']:
        hidden_feat_repr_dims = int(opt['time'] // opt['step_size']) * [opt['hidden_dim']]
        feat_repr_dims = [dataset.data.x.shape[1]] + hidden_feat_repr_dims + [dataset.num_classes]
        model = GNNMLP(opt, dataset, device, feat_repr_dims,
                       enable_mlp=True if opt['function'] == 'mlp' else False,
                       enable_gcn=True if opt['function'] in ['gcn2', 'gcn_dgl', 'gcn_res_dgl'] else False,
                       learnable_mixing=False, use_sage=False, use_gat=False, gat_num_heads=1,
                       top_is_proj=False, use_prelu=False, dropout=opt['dropout']).to(device)
    elif opt['function'] in ['gat']:
        model = GAT(opt, dataset, hidden=[opt['hidden_dim']], dropout=opt['dropout'], device=device).to(device)
        # hidden_feat_repr_dims = int(opt['time'] // opt['step_size']) * [opt['hidden_dim']]
        # feat_repr_dims = [dataset.data.x.shape[1]] + hidden_feat_repr_dims + [dataset.num_classes]
        # model = GNNMLP(opt, dataset, device, feat_repr_dims,
        #                enable_mlp=True if opt['function'] == 'mlp' else False,
        #                enable_gcn=True, #required for GAT
        #                learnable_mixing=False, use_sage=False,
        #                use_gat=True if opt['function'] == 'gat' else False, gat_num_heads=1,
        #                top_is_proj=False, use_prelu=False, dropout=opt['dropout']).to(device)
    else:
        model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)

    # num_params = model.parameters().numel()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    runs = []
    for i in range(100):
        start = time.time()
        _ = model(feat, pos_encoding)
        run_time = time.time() - start
        runs.append(run_time)
    av_fwd = np.mean(runs)
    std_fwd = np.std(runs)
    wandb_results = {"num_params": num_params,"av_fwd": av_fwd, "std_fwd": std_fwd}
    wandb.log(wandb_results)
    wandb_run.finish()
    print(f"function {opt['function']}: num_params {num_params}, av_fwd {av_fwd}, std_fwd {std_fwd}")

from GNN_GGCN import GGCN #todo merge this into other function
def wall_clock_ablation_GGCN(opt):
    if 'wandb_run_name' in opt.keys():
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
    else:
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               reinit=True, config=opt, allow_val_change=True)  # required when update config

    opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!
    if opt['gcn_params']: #temp function for GCN ablation
        unpack_gcn_params(opt)
    if opt['greed_params']: #temp function for GCN ablation
        unpack_greed_params(opt)

    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    if opt['dataset'] in ['chameleon','squirrel','other hetero?']:
        dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    if opt['geom_gcn_splits']:
        if opt['dataset'] == "Citeseer":
            dataset = get_dataset(opt, '../data', opt['not_lcc']) #geom-gcn citeseer uses splits over LCC and not_LCC so need to repload each split
        data = get_fixed_splits(dataset.data, opt['dataset'], 0)
        dataset.data = data

    data = dataset.data
    feat = data.x
    pos_encoding = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = feat
    num_labels = dataset.num_classes
    adj = to_dense_adj(data.edge_index).squeeze()
    #for sparse
    #     n = data.x.shape[0]
    #     e = data.edge_index.shape[1]
    #     adj = torch.sparse_coo_tensor(data.edge_index, e*[1], (n, n))
    #     #10138 / 2485
    #example params provided by GGCN
    use_degree = True
    use_sign = True
    use_decay = True
    use_bn = False
    use_ln = False
    model = GGCN(nfeat=features.shape[1], nlayers=2, nhidden=opt['hidden_dim'], nclass=num_labels, dropout=0.0,
             decay_rate=0.0, exponent=3.0, use_degree=use_degree, use_sign=use_sign,
             use_decay=use_decay, use_sparse=True, scale_init=0.5,
             deg_intercept_init=0.5, use_bn=use_bn, use_ln=use_ln).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    runs = []
    for i in range(100):
        start = time.time()
        # _ = model(feat, pos_encoding)
        output = model(features,adj)

        run_time = time.time() - start
        runs.append(run_time)
    av_fwd = np.mean(runs)
    std_fwd = np.std(runs)
    wandb_results = {"num_params": num_params,"av_fwd": av_fwd, "std_fwd": std_fwd}
    wandb.log(wandb_results)
    wandb_run.finish()
    print(f"function {opt['function']}: num_params {num_params}, av_fwd {av_fwd}, std_fwd {std_fwd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from '
                             'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                        help='the number of splits to repeat the results on')
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
    parser.add_argument('--patience', type=int, default=None, help='set if training should use patience on val acc')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, help='laplacian, transformer, greed, GAT')
    # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
    #                     help='Add a fully connected layer to the encoder.')
    # parser.add_argument('--add_source', dest='add_source', action='store_true',
    #                     help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=0.1,
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
    # parser.add_argument("--no_early", action="store_true",
    #                     help="Whether or not to use early stopping of the ODE integrator when testing.")
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
    parser.add_argument('--attention_type', type=str, default="",
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
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                        help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                        help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                        help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    # parser.add_argument('--symmetric_QK', action='store_true',
    #                     help='makes the attention symmetric for rewring in QK space')
    # parser.add_argument('--symmetric_attention', action='store_true',
    #                     help='makes the attention symmetric via (A+A.T)/2')#for rewring in QK space')
    # parser.add_argument('--sym_row_max', action='store_true',
    #                     help='makes every row sum less than 1 by dividing by max rum some')
    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")
    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

    # wandb logging and tuning
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('-wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="graph_neural_diffusion", type=str,
                        help="jrowbottomwnb, ger__man")  # not used as default set in web browser settings
    parser.add_argument('--wandb_project', default="greed", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    # parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    # replaces the above
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 254],
                        help='list of epochs to log gradient flow, 1 based')
    parser.add_argument('--run_track_reports', action='store_true', help="run_track_reports")
    parser.add_argument('--save_wandb_reports', action='store_true', help="save_wandb_reports")
    parser.add_argument('--save_local_reports', action='store_true', help="save_local_reports")

    # wandb setup sweep args
    parser.add_argument('--tau_reg', type=float, default=2)
    parser.add_argument('--test_mu_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_no_chanel_mix', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric_L', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric_R', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_remove_tanh', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_symmetric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_outside', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_linear_L0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_R1R2_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_grand_metric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_ones', type=str, default='True')  # action='store_true')

    # Temp changing these to be strings so can tune over
    parser.add_argument('--use_mlp', type=str, default='False')  # action='store_true')
    parser.add_argument('--add_source', type=str, default='False',
                        help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--no_early', type=str, default='False')  # action='store_true')
    parser.add_argument('--symmetric_QK', type=str, default='False',
                        help='makes the attention symmetric for rewring in QK space')
    parser.add_argument('--symmetric_attention', type=str, default='False',
                        help='makes the attention symmetric via (A+A.T)/2')  # for rewring in QK space')
    parser.add_argument('--sym_row_max', type=str, default='False',
                        help='makes every row sum less than 1 by dividing by max rum some')

    # greed args
    parser.add_argument('--use_best_params', action='store_true', help="flag to take the best BLEND params")
    parser.add_argument('--greed_momentum', action='store_true', help="flag to use momentum grad flow")
    parser.add_argument('--momentum_alpha', type=float, default=0.2, help="alpha to use in momentum grad flow")
    parser.add_argument('--dim_p_omega', type=int, default=16, help="inner dimension for Omega")
    parser.add_argument('--dim_p_w', type=int, default=16, help="inner dimension for W")
    parser.add_argument('--gamma_epsilon', type=float, default=0.01,
                        help="epsilon value used for numerical stability in get_gamma")

    parser.add_argument('--XN_no_activation', type=str, default='False',
                        help='whether to relu activate the terminal state')
    parser.add_argument('--m2_mlp', type=str, default='False', help='whether to use decoder mlp')
    parser.add_argument('--attention_activation', type=str, default='exponential',
                        help='[exponential, sigmoid] activations for the GRAM matrix')
    parser.add_argument('--attention_normalisation', type=str, default='sym_row_col',
                        help='[mat_row_max, sym_row_col, row_bottom, "best"] how to normalise')
    parser.add_argument('--T0term_normalisation', type=str, default='T0_identity',
                        help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
    parser.add_argument('--laplacian_norm', type=str, default='lap_noNorm',
                        help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')
    parser.add_argument('--R_T0term_normalisation', type=str, default='T0_identity',
                        help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
    parser.add_argument('--R_laplacian_norm', type=str, default='lap_noNorm',
                        help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')

    parser.add_argument('--alpha_style', type=str, default='', help='"sigmoid", "free", "forced", "matrix"')
    parser.add_argument('--fix_alpha', type=float, default=None, help='control balance between diffusion and repulsion')
    parser.add_argument('--diffusion', type=str, default='True', help='turns on diffusion')
    parser.add_argument('--repulsion', type=str, default='False', help='turns on repulsion')
    # parser.add_argument('--drift', type=str, default='False', help='turns on drift')
    parser.add_argument('--W_type', type=str, default='identity', help='identity, diag, full')
    parser.add_argument('--R_W_type', type=str, default='identity', help='for repulsion: identity, diag, full')
    parser.add_argument('--R_depon_A', type=str, default='', help='R dependancy in A')
    parser.add_argument('--W_beta', type=float, default=0.5, help='for cgnn Ws orthoganal update')
    parser.add_argument('--tau_residual', type=str, default='False', help='makes tau residual')

    parser.add_argument('--drift', type=str, default='False', help='turns on drift')
    parser.add_argument('--gnl_thresholding', type=str, default='False', help='turns on pseudo inverse thresholding')
    parser.add_argument('--lie_trotter', type=str, default=None, help='None, gen_0, gen_1, gen_2')

    # GCN ablation args
    parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
    parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
    parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
    parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
    parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')
    parser.add_argument('--gcn_mid_dropout', type=str, default='False', help='dropout between GCN layers')
    parser.add_argument('--gcn_params', nargs='+', default=None, help='list of args for gcn ablation')
    parser.add_argument('--gcn_params_idx', type=int, default=0, help='index to track GCN ablation')

    # greed non linear args
    parser.add_argument('--gnl_style', type=str, default='scaled_dot',
                        help='scaled_dot, softmax_attention, general_graph')
    parser.add_argument('--gnl_activation', type=str, default='identity', help='identity, sigmoid, ...')
    parser.add_argument('--gnl_measure', type=str, default='ones', help='ones, deg_poly, nodewise')
    parser.add_argument('--gnl_omega', type=str, default='zero', help='zero, diag, sum')
    parser.add_argument('--gnl_omega_diag', type=str, default='free', help='free, const')
    parser.add_argument('--gnl_omega_activation', type=str, default='identity', help='identity, exponential')
    parser.add_argument('--gnl_W_style', type=str, default='sum', help='sum, prod, GS, cgnn, diag_dom')
    parser.add_argument('--gnl_attention', type=str, default='False', help='turns on attention for gnl general graph')

    parser.add_argument('--k_blocks', type=int, default=5, help='k_blocks')
    parser.add_argument('--block_size', type=int, default=5, help='block_size')
    parser.add_argument('--k_diags', type=float, default=11, help='k_diags')
    parser.add_argument('--k_diag_pc', type=float, default=0.1, help='percentage or dims diagonal')
    parser.add_argument('--gnl_omega_params', nargs='+', default=None, help='list of Omega args for ablation')
    parser.add_argument('--gnl_W_params', nargs='+', default=None, help='list of W args for ablation')
    parser.add_argument('--gnl_W_diag_init', type=str, default='identity',
                        help='init of diag elements [identity, uniform, linear]')
    parser.add_argument('--gnl_W_param_free', type=str, default='True', help='allow parameter to require gradient')
    parser.add_argument('--gnl_W_diag_init_q', type=float, default=1.0, help='slope of init of spectrum of W')
    parser.add_argument('--gnl_W_diag_init_r', type=float, default=0.0, help='intercept of init of spectrum of W')
    parser.add_argument('--two_hops', type=str, default='False', help='flag for 2-hop energy')
    parser.add_argument('--time_dep_w', type=str, default='False', help='Learn a time dependent potentials')
    parser.add_argument('--time_dep_struct_w', type=str, default='False',
                        help='Learn a structured time dependent potentials')
    parser.add_argument('--target_homoph', type=str, default='0.80',
                        help='target_homoph for syn_cora [0.00,0.10,..,1.00]')
    parser.add_argument('--greed_params', nargs='+', default=None, help='list of args for focus models')

    args = parser.parse_args()
    opt = vars(args)

    if opt['function'] in ['greed', 'greed_scaledDP', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero',
                           'greed_non_linear', 'greed_lie_trotter', 'gcn', 'gcn2', 'mlp', 'gcn_dgl', 'gcn_res_dgl',
                           'gat', 'GAT', 'GGCN']:
        opt = greed_run_params(opt)  ###basic params for GREED

    if not opt['wandb_sweep']:  # sweeps are run from YAML config so don't need these
        opt = not_sweep_args(opt, project_name='greed_runs', group_name='testing')
        # this includes args for running locally - specified in YAML for tunes
        #  opt = greed_hyper_params(opt)
        # opt = greed_ablation_params(opt)

    # applied to both sweeps and not sweeps
    opt = tf_ablation_args(opt)
    if opt['function'] == "GGCN":
        wall_clock_ablation_GGCN(opt)
    else:
        wall_clock_ablation(opt)

