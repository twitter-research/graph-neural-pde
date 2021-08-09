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
from best_params import best_params_dict
from run_GNN import print_model_params, get_optimizer, test, test_OGB, train, get_label_masks, add_labels


def av_degree():
    datas = ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS', 'Computers', 'Photo']

    for data in datas:
        print(data)
        dataset = get_dataset(opt, '../data', opt['not_lcc'])
        print(f"data_num_edges {dataset.data.edge_index.shape[1]}")
        print(f"data_num_nodes {dataset.data.y.shape[0]}")


def main(opt):
    meta_dict = {}
    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    meta_dict['data_num_edges'] = dataset.data.edge_index.shape[1]
    meta_dict['data_num_nodes'] = dataset.data.y.shape[0]

    if opt['beltrami']:
        pos_encoding = apply_beltrami(dataset.data, opt).to(device)
        opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
        pos_encoding = None

    model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)

    if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

    data = dataset.data.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = train_acc = val_acc = test_acc = 0

    patience_counter = 0
    patience = 100
    prev_fwd_nfe = 0
    prev_back_nfe = 0
    fwd_time = 0
    back_time = 0
    for epoch in range(1, opt['epoch']):
        if patience_counter == patience:
            meta_dict['last_epoch'] = {'epoch': epoch, 'fwd_nfe': prev_fwd_nfe - prev2_fwd_nfe, 'back_nfe': prev_back_nfe - prev2_back_nfe,
                                       'fwd_time': fwd_time, 'back_time': back_time}
            break

        start_time = time.time()
        this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
        loss, fwd_time, back_time = train(model, optimizer, data, pos_encoding)

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
        if epoch in opt['epoch_snapshots']:
            meta_dict[epoch] = {'epoch': epoch, 'fwd_nfe':model.fm.sum - prev_fwd_nfe, 'back_nfe': model.bm.sum - prev_back_nfe,
                                'fwd_time': fwd_time, 'back_time': back_time}

        prev2_fwd_nfe = prev_fwd_nfe
        prev2_back_nfe = prev_back_nfe
        prev_fwd_nfe = model.fm.sum
        prev_back_nfe = model.bm.sum


    if epoch != opt['epoch']:
        meta_dict['last_epoch'] = {'epoch': epoch, 'fwd_nfe': prev_fwd_nfe - prev2_fwd_nfe,
                                   'back_nfe': prev_back_nfe - prev2_back_nfe,
                                   'fwd_time': fwd_time, 'back_time': back_time}

    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(val_acc, test_acc, best_epoch))
    return train_acc, val_acc, test_acc, meta_dict


def train(model, optimizer, data, pos_encoding=None):
    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    start_fwd = time.time()
    out = model(feat, pos_encoding)
    fwd_time = time.time() - start_fwd

    if model.opt['dataset'] == 'ogbn-arxiv':
        lf = torch.nn.functional.nll_loss
        loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        lf = torch.nn.CrossEntropyLoss()
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        regularization_coeffs = model.regularization_coeffs

        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )
        loss = loss + reg_loss

    model.fm.update(model.getNFE())
    model.resetNFE()
    start_back = time.time()
    loss.backward()
    back_time = time.time() - start_back

    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item(), fwd_time, back_time


def ODE_solver_ablation(cmd_opt):
    datas = ['Cora', 'Citeseer' ,'Pubmed','CoauthorCS','Computers','Photo']
    # datas = ['Pubmed','CoauthorCS','Computers','Photo']
    steps = [1.0, 2.0, 4.0, 8.0]
    methods = ['best', 'euler']

    rows = []
    for i, ds in enumerate(datas):
        best_opt = best_params_dict[ds]
        opt = {**cmd_opt, **best_opt}

        opt['no_early'] = True  # no implementation of early stop solver for explicit euler

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

                for it in range(opt['ablation_its']):
                    print(f"Running Best Params for {ds}")
                    train_acc, val_acc, test_acc, meta_dict = main(opt)
                    row = [ds, opt['time'], it, opt['method'], opt['step_size'], opt['adjoint_method'],
                           opt['adjoint_step_size'],
                           meta_dict[1]['epoch'], meta_dict[1]['fwd_nfe'], meta_dict[1]['back_nfe'],
                           meta_dict[1]['fwd_time'], meta_dict[1]['back_time'],
                           meta_dict[11]['epoch'], meta_dict[11]['fwd_nfe'], meta_dict[11]['back_nfe'],
                           meta_dict[11]['fwd_time'], meta_dict[11]['back_time'],
                           opt['epoch'],
                           meta_dict['last_epoch']['epoch'], meta_dict['last_epoch']['fwd_nfe'],
                           meta_dict['last_epoch']['back_nfe'], meta_dict['last_epoch']['fwd_time'],
                           meta_dict['last_epoch']['back_time'],
                           train_acc, val_acc, test_acc, ]
                    rows.append(row)

            elif method == 'euler':
                for step in steps:
                    opt['method'] = 'euler'
                    opt['step_size'] = step
                    opt['adjoint_method'] = 'euler'
                    opt['adjoint_step_size'] = step

                    for it in range(opt['ablation_its']):
                        print(f"Running Best Params for {ds}")
                        train_acc, val_acc, test_acc, meta_dict = main(opt)
                        # row = [ds, opt['time'], it, opt['method'], opt['step_size'], opt['adjoint_method'],
                        #        opt['adjoint_step_size'], train_acc, val_acc, test_acc]
                        row = [ds, opt['time'], it, opt['method'], opt['step_size'], opt['adjoint_method'],
                               opt['adjoint_step_size'],
                               meta_dict[1]['epoch'], meta_dict[1]['fwd_nfe'], meta_dict[1]['back_nfe'],
                               meta_dict[1]['fwd_time'], meta_dict[1]['back_time'],
                               meta_dict[11]['epoch'], meta_dict[11]['fwd_nfe'], meta_dict[11]['back_nfe'],
                               meta_dict[11]['fwd_time'], meta_dict[11]['back_time'],
                               opt['epoch'],
                               meta_dict['last_epoch']['epoch'], meta_dict['last_epoch']['fwd_nfe'],
                               meta_dict['last_epoch']['back_nfe'], meta_dict['last_epoch']['fwd_time'],
                               meta_dict['last_epoch']['back_time'],
                               train_acc, val_acc, test_acc, ]

                        rows.append(row)

        df = pd.DataFrame(rows, columns=['dataset', 'time', 'iteration', 'method', 'step_size', 'adjoint_method','adjoint_step_size',
                                         'epoch1','epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
                                         'epoch11', 'epoch11_fwd_nfe', 'epoch11_back_nfe', 'epoch11_fwd_time','epoch11_back_time',
                                         'max_epoch',
                                         'epochlast', 'epochlast_fwd_nfe', 'epochlast_back_nfe', 'epochlast_fwd_time','epochlast_back_time',
                                         'train_acc', 'val_acc', 'test_acc'])

        pd.set_option('display.max_columns', None)

        mean_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'],
                                    index=['dataset', 'time', 'method', 'step_size'],
                                    aggfunc=np.mean,
                                    margins=True)

        mean_table_details = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc',
                                                        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
                                                        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
                                                        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time'],
                                    index=['dataset', 'time', 'method', 'step_size','epoch1','epoch11','max_epoch'],
                                    aggfunc=np.mean,
                                    margins=True)


        mean_table_details = mean_table_details.reindex(labels=[
        # 'dataset','time','method','step_size','epoch1','epoch11','max_epoch',
        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time',
        'train_acc','val_acc','test_acc'], axis=1)

        std_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'],
                                   index=['dataset', 'time', 'method', 'step_size'],
                                   aggfunc=np.std, margins=True)

        std_table_details = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc',
                                                        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
                                                        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
                                                        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time'],
                                    index=['dataset', 'time', 'method', 'step_size','epoch1','epoch11','max_epoch'],
                                    aggfunc=np.std,
                                    margins=True)

        std_table_details = std_table_details.reindex(labels=[
        # 'dataset','time','method','step_size','epoch1','epoch11','max_epoch',
        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time',
        'train_acc','val_acc','test_acc'], axis=1)

        df.to_csv(f"../ablations/ODE_solver_data_{ds}.csv")
        mean_table.to_csv(f"../ablations/ODE_solver_mean_{ds}.csv")
        mean_table_details.to_csv(f"../ablations/ODE_solver_mean_{ds}_details.csv")
        std_table.to_csv(f"../ablations/ODE_solver_std_{ds}.csv")
        std_table_details.to_csv(f"../ablations/ODE_solver_std_{ds}_details.csv")

        print(df)
        print(mean_table)
        print(std_table)


def attention_ablation(cmd_opt):
    datas = ['Cora' ,'Citeseer','Pubmed','CoauthorCS','Computers','Photo']
    attentions = ['scaled_dot', 'cosine_sim', 'pearson', 'exp_kernel']

    rows = []
    for i, ds in enumerate(datas):
        best_opt = best_params_dict[ds]
        opt = {**cmd_opt, **best_opt}

        hp_attention = opt['attention_type']
        for attention in attentions:
            opt['attention_type'] = attention
            for it in range(opt['ablation_its']):
                print(f"Running Best Params for {ds}")
                train_acc, val_acc, test_acc, meta_dict = main(opt)
                row = [ds, it, hp_attention, opt['attention_type'], train_acc, val_acc, test_acc]
                rows.append(row)

        df = pd.DataFrame(rows, columns=['dataset', 'iteration', 'hp_attention', 'attention_type', 'train_acc', 'val_acc', 'test_acc'])
        pd.set_option('display.max_columns', None)
        df.to_csv(f"../ablations/attention_data_{ds}.csv")

        mean_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'],
                                    index=["dataset", 'hp_attention', "attention_type"],
                                    aggfunc={'train_acc': np.mean, 'val_acc': np.mean, 'test_acc': np.mean},
                                    margins=True)
        mean_table.to_csv(f"../ablations/attention_mean_{ds}.csv")

        std_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc'], index=["dataset", 'hp_attention', "attention_type"],
                                   aggfunc={'train_acc': np.std, 'val_acc': np.std, 'test_acc': np.std}, margins=True)
        std_table.to_csv(f"../ablations/attention_std_{ds}.csv")

        print(df)
        print(mean_table)
        print(std_table)


def runtime_ablation(cmd_opt):
    # datas = ['Cora', 'Citeseer', 'Pubmed','CoauthorCS','Computers','Photo']
    datas = ['Computers','Photo']
    methods = ['BLEND', 'BLEND_kNN']

    knn_dict = {'Cora':
                    {'rewiring': 'gdc', 'gdc_method': 'ppr', 'ppr_alpha': 0.15,
                      'gdc_sparsification': 'topk', 'gdc_k': 32},
                'Citeseer':
                    {'rewiring': 'gdc', 'gdc_method': 'ppr', 'ppr_alpha': 0.15,
                      'gdc_sparsification': 'topk', 'gdc_k': 64},
                'Pubmed':
                    {'rewiring': 'gdc', 'gdc_method': 'ppr', 'ppr_alpha': 0.15,
                    'gdc_sparsification': 'threshold', 'gdc_threshold': 0.00037260592724232223},
                'CoauthorCS':
                    {'rewiring': 'gdc', 'gdc_method': 'ppr', 'ppr_alpha': 0.15,
                    'gdc_sparsification': 'threshold', 'gdc_threshold': 0.002608137444765515},
                'Computers':
                    {'rewiring': 'gdc', 'gdc_method': 'ppr', 'ppr_alpha': 0.15,
                    'gdc_sparsification': 'threshold', 'gdc_threshold': 0.00037260592724232223},
                'Photo':
                    {'rewiring': 'gdc', 'gdc_method': 'ppr', 'ppr_alpha': 0.15,
                    'gdc_sparsification': 'threshold', 'gdc_threshold': 0.0002589973811782757}}

    rows = []
    for i, ds in enumerate(datas):
        best_opt = best_params_dict[ds]
        opt = {**cmd_opt, **best_opt}

        opt['no_early'] = True  # no implementation of early stop solver for explicit euler //also not a neccessary comparison against GAT
        opt['epoch'] = 100
        opt['ablation_its'] = 2
        opt['self_loop_weight'] = 1.0

        for method in methods:
            if method == 'BLEND':
                for it in range(opt['ablation_its']):
                    total_time_start = time.time()
                    print(f"Running Best Params for {ds}")
                    train_acc, val_acc, test_acc, meta_dict = main(opt)
                    row = [ds, opt['time'], it, method, opt['step_size'], opt['adjoint_method'], opt['adjoint_step_size'],
                           meta_dict['data_num_edges'], meta_dict['data_num_nodes'],
                           meta_dict[1]['epoch'], meta_dict[1]['fwd_nfe'], meta_dict[1]['back_nfe'],
                           meta_dict[1]['fwd_time'], meta_dict[1]['back_time'],
                           meta_dict[11]['epoch'], meta_dict[11]['fwd_nfe'], meta_dict[11]['back_nfe'],
                           meta_dict[11]['fwd_time'], meta_dict[11]['back_time'],
                           opt['epoch'],
                           meta_dict['last_epoch']['epoch'], meta_dict['last_epoch']['fwd_nfe'],
                           meta_dict['last_epoch']['back_nfe'], meta_dict['last_epoch']['fwd_time'],
                           meta_dict['last_epoch']['back_time'],
                           train_acc, val_acc, test_acc, time.time() - total_time_start]
                    rows.append(row)

            elif method == 'BLEND_kNN':
                opt['rewiring'] = 'gdc'
                opt['gdc_method'] = 'ppr'
                opt['ppr_alpha'] = 0.15
                opt['gdc_sparsification'] = knn_dict[ds]['gdc_sparsification']
                if knn_dict[ds]['gdc_sparsification'] == 'threshold':
                    opt['gdc_threshold'] = knn_dict[ds]['gdc_threshold']
                elif knn_dict[ds]['gdc_sparsification'] == 'topk':
                    opt['gdc_k'] = knn_dict[ds]['gdc_k']

                for it in range(opt['ablation_its']):
                    total_time_start = time.time()
                    print(f"Running Best Params for {ds}")
                    train_acc, val_acc, test_acc, meta_dict = main(opt)
                    row = [ds, opt['time'], it, method, opt['step_size'], opt['adjoint_method'], opt['adjoint_step_size'],
                           meta_dict['data_num_edges'], meta_dict['data_num_nodes'],
                           meta_dict[1]['epoch'], meta_dict[1]['fwd_nfe'], meta_dict[1]['back_nfe'],
                           meta_dict[1]['fwd_time'], meta_dict[1]['back_time'],
                           meta_dict[11]['epoch'], meta_dict[11]['fwd_nfe'], meta_dict[11]['back_nfe'],
                           meta_dict[11]['fwd_time'], meta_dict[11]['back_time'],
                           opt['epoch'],
                           meta_dict['last_epoch']['epoch'], meta_dict['last_epoch']['fwd_nfe'],
                           meta_dict['last_epoch']['back_nfe'], meta_dict['last_epoch']['fwd_time'],
                           meta_dict['last_epoch']['back_time'],
                           train_acc, val_acc, test_acc, time.time() - total_time_start]
                    rows.append(row)


        df = pd.DataFrame(rows, columns=['dataset', 'time', 'iteration', 'method', 'step_size', 'adjoint_method','adjoint_step_size',
                                         'data_num_edges', 'data_num_nodes',
                                         'epoch1','epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
                                         'epoch11', 'epoch11_fwd_nfe', 'epoch11_back_nfe', 'epoch11_fwd_time','epoch11_back_time',
                                         'max_epoch',
                                         'epochlast', 'epochlast_fwd_nfe', 'epochlast_back_nfe', 'epochlast_fwd_time','epochlast_back_time',
                                         'train_acc', 'val_acc', 'test_acc','total_time'])

        pd.set_option('display.max_columns', None)

        mean_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc', 'total_time'],
                                    index=['dataset', 'time', 'method', 'step_size', 'data_num_edges', 'data_num_nodes'],
                                    aggfunc=np.mean,
                                    margins=True)

        mean_table_details = pd.pivot_table(df, values=['data_num_edges', 'data_num_nodes', 'train_acc', 'val_acc', 'test_acc', 'total_time',
                                                        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
                                                        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
                                                        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time'],
                                    index=['dataset', 'time', 'method', 'step_size','epoch1','epoch11','max_epoch'],
                                    aggfunc=np.mean,
                                    margins=True)

        mean_table_details = mean_table_details.reindex(labels=['data_num_edges', 'data_num_nodes',
        # 'dataset','time','method','step_size','epoch1','epoch11','max_epoch',
        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time',
        'train_acc','val_acc','test_acc','total_time'], axis=1)

        std_table = pd.pivot_table(df, values=['train_acc', 'val_acc', 'test_acc', 'total_time'],
                                   index=['dataset', 'time', 'method', 'step_size'],
                                   aggfunc=np.std, margins=True)

        std_table_details = pd.pivot_table(df, values=['data_num_edges', 'data_num_nodes', 'train_acc', 'val_acc', 'test_acc', 'total_time',
                                                        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
                                                        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
                                                        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time'],
                                    index=['dataset', 'time', 'method', 'step_size','epoch1','epoch11','max_epoch'],
                                    aggfunc=np.std,
                                    margins=True)

        std_table_details = std_table_details.reindex(labels=['data_num_edges', 'data_num_nodes',
        # 'dataset','time','method','step_size','epoch1','epoch11','max_epoch',
        'epoch1_fwd_nfe','epoch1_back_nfe','epoch1_fwd_time','epoch1_back_time',
        'epoch11_fwd_nfe','epoch11_back_nfe','epoch11_fwd_time','epoch11_back_time',
        'epochlast','epochlast_fwd_nfe','epochlast_back_nfe','epochlast_fwd_time','epochlast_back_time',
        'train_acc','val_acc','test_acc','total_time'], axis=1)

        df.to_csv(f"../ablations/run_time_data_{ds}.csv")
        mean_table.to_csv(f"../ablations/run_time_mean_{ds}.csv")
        mean_table_details.to_csv(f"../ablations/run_time_mean_{ds}_details.csv")
        std_table.to_csv(f"../ablations/run_time_std_{ds}.csv")
        std_table_details.to_csv(f"../ablations/run_time_std_{ds}_details.csv")

        print(df)
        print(mean_table)
        print(std_table)


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

    # ablation args
    parser.add_argument('--ablation_its', type=int, default=8, help="number of iterations to average over")

    def arg_as_list(s):
        import ast
        v = ast.literal_eval(s)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
        return v

    parser.add_argument("--epoch_snapshots", type=arg_as_list, default='[1,11]', help="List of values")

    args = parser.parse_args()
    opt = vars(args)

    # ODE_solver_ablation(opt)
    # attention_ablation(opt)

    # runtime_ablation(opt)
    av_degree()