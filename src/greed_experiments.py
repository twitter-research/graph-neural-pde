import datetime, time
import torch
import numpy as np
import pandas as pd
from run_GNN import main
from greed_params import greed_run_params, not_sweep_args, tf_ablation_args, default_params
from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from GNN_GCN import GCN, MLP
from GNN_GCNMLP import GNNMLP
from data import get_dataset, set_train_val_test_split

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

def wall_clock_ablation():
    opt = default_params()
    #load Cora
    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    data = dataset.data
    feat = data.x
    pos_encoding = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows = []
    for name in ['gcn', 'diag', 'diag_dom']:#, 'ggcn', 'sheaf']:
        for hd in [16, 32, 64, 128]:
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
                               top_is_proj=False, use_prelu=False, dropout=opt['dropout']
                               ).to(device)
            else:
                model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(
                    device)

            runs = []
            for i in range(100):
                start = time.time()
                out = model(feat, pos_encoding)
                run_time = time.time() - start
                runs.append(run_time)

            rows.append(name, hd, np.mean(runs), np.std(runs))
    df = pd.DataFrame(rows, columns=['model, hidden_dim, av_fwd, std_fwd'])
    # df.to_csv("../ablations/run_time.csv")

if __name__ == "__main__":
    run_track_flow_experiments()