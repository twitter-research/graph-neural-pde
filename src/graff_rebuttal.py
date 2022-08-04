import numpy as np
import pandas as pd

def best_gcn_time_step(path, name):

    df = pd.read_csv(path + name)
    all_datas = ['Cora', 'Citeseer', 'chameleon', 'squirrel']
    gcn_models = [0, 1, 2, 3, 4, 5]
    graff_models = [('diag_dom', True), ('diag_dom', False), ('diag', True), ('diag', False)]
    times = [2, 3, 4]
    steps = [0.5, 1.0]
    best_idxs = []
    sweeps = ['fldv2pwo', 'b5ve7b4e', 'huykgmkd']
    for sw in ['fldv2pwo', 'huykgmkd']:
        datas = all_datas[:-1] if sw == 'fldv2pwo' else [all_datas[-1]]
        for d in datas:
            for m in gcn_models:
                for t in times:
                    for s in steps:
                        temp_df = df[(df['Sweep']==sw) &(df['dataset']==d) & (df['gcn_params_idx']==m) & (df['time']==t) & (df['step_size']==s)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    for sw in ['b5ve7b4e']:
        for d in all_datas:
            for m, nl in graff_models:
                for t in times:
                    for s in steps:
                        temp_df = df[(df['Sweep']==sw) &(df['dataset']==d) & (df['gnl_W_style']==m) & (df['pointwise_nonlin']==nl)
                                     & (df['time']==t) & (df['step_size']==s)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)


    best_df = df.iloc[best_idxs]

    best_df.to_csv(path + "best_time_step.csv")
    print(best_df)

def best_gcn(path, name):

    df = pd.read_csv(path + name)
    all_datas = ['Cora', 'Citeseer', 'chameleon', 'squirrel']
    gcn_models = [0, 1, 2, 3, 4, 5]
    graff_models = [('diag_dom', True), ('diag_dom', False), ('diag', True), ('diag', False)]
    times = [2, 3, 4]
    steps = [0.5, 1.0]
    best_idxs = []
    sweeps = ['fldv2pwo', 'b5ve7b4e', 'huykgmkd']
    for sw in ['fldv2pwo', 'huykgmkd']:
        datas = all_datas[:-1] if sw == 'fldv2pwo' else [all_datas[-1]]
        for d in datas:
            for m in gcn_models:
                        temp_df = df[(df['Sweep']==sw) &(df['dataset']==d) & (df['gcn_params_idx']==m)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    for sw in ['b5ve7b4e']:
        for d in all_datas:
            for m, nl in graff_models:
                        temp_df = df[(df['Sweep']==sw) &(df['dataset']==d) & (df['gnl_W_style']==m) & (df['pointwise_nonlin']==nl)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    best_df = df.iloc[best_idxs]

    best_df.to_csv(path + "best_gcn.csv")
    print(best_df)

def best_syn_cora_time_step(path, name):

    df = pd.read_csv(path + name)
    all_datas = np.arange(0,1.1,0.1).tolist() #['0.00','0.10','0.20','0.30','0.40','0.50','0.60','0.70','0.80','0.90','1.00']
    gcn_models = [0, 1, 2, 3, 4, 5]
    graff_models = [('diag_dom', True), ('diag_dom', False), ('diag', True), ('diag', False)]
    times = [2, 3, 4]
    steps = [0.5, 1.0]
    best_idxs = []
    sweeps = ['mysqbj5j', '20r3vs4h']
    for sw in ['mysqbj5j']:
        for d in all_datas:
            for m in gcn_models:
                for t in times:
                    for s in steps:
                        temp_df = df[(df['Sweep']==sw) &(df['target_homoph']==round(d,2)) & (df['gcn_params_idx']==m) & (df['time']==t) & (df['step_size']==s)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    for sw in ['20r3vs4h']:
        for d in all_datas:
            for m, nl in graff_models:
                for t in times:
                    for s in steps:
                        temp_df = df[(df['Sweep']==sw) &(df['target_homoph']==round(d,2)) & (df['gnl_W_style']==m) & (df['pointwise_nonlin']==nl)
                                     & (df['time']==t) & (df['step_size']==s)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)


    best_df = df.iloc[best_idxs]

    best_df.to_csv(path + "best_syn_cora_time_step.csv")
    print(best_df)

def best_syn_cora(path, name):

    df = pd.read_csv(path + name)
    all_datas = np.arange(0,1.1,0.1).tolist()
    gcn_models = [0, 1, 2, 3, 4, 5]
    graff_models = [('diag_dom', True), ('diag_dom', False), ('diag', True), ('diag', False)]
    times = [2, 3, 4]
    steps = [0.5, 1.0]
    best_idxs = []
    sweeps = ['mysqbj5j', '20r3vs4h']
    for sw in ['mysqbj5j']:
        for d in all_datas:
            for m in gcn_models:
                        temp_df = df[(df['Sweep']==sw) &(df['target_homoph']==round(d,2)) & (df['gcn_params_idx']==m)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    for sw in ['20r3vs4h']:
        for d in all_datas:
            for m, nl in graff_models:
                        temp_df = df[(df['Sweep']==sw) &(df['target_homoph']==round(d,2)) & (df['gnl_W_style']==m) & (df['pointwise_nonlin']==nl)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    best_df = df.iloc[best_idxs]

    best_df.to_csv(path + "best_syn_cora.csv")
    print(best_df)

if __name__ == "__main__":
    path = "../ablations/"
    name = "rebuttal_gcn_ablation.csv"
    # best_gcn_time_step(path, name)
    # best_gcn(path, name)

    name = "rebuttal_syn_cora.csv"
    best_syn_cora_time_step(path, name)
    best_syn_cora(path, name)
