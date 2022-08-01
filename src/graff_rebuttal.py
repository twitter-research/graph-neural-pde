
import pandas as pd

def import_csv(path, name):

    df = pd.read_csv(path + name)
    datas = ['Cora', 'Citeseer', 'chameleon', 'squirrel']
    gcn_models = [0, 1, 2, 3, 4, 5]
    graff_models = [('diag_dom', True), ('diag_dom', False), ('diag', True), ('diag', False)]
    times = [2, 3, 4]
    steps = [0.5, 1.0]
    best_idxs = []
    sweeps = ['fldv2pwo', 'b5ve7b4e']
    for sw in ['fldv2pwo']:
        for d in datas:
            for m in gcn_models:
                for t in times:
                    for s in steps:
                        temp_df = df[(df['Sweep']==sw) &(df['dataset']==d) & (df['gcn_params_idx']==m) & (df['time']==t) & (df['step_size']==s)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)

    for sw in ['b5ve7b4e']:
        for d in datas:
            for m, nl in graff_models:
                for t in times:
                    for s in steps:
                        temp_df = df[(df['Sweep']==sw) &(df['dataset']==d) & (df['gnl_W_style']==m) & (df['pointwise_nonlin']==nl)
                                     & (df['time']==t) & (df['step_size']==s)]
                        best_idx = temp_df['test_mean'].idxmax(axis=0)
                        best_idxs.append(best_idx)


    best_df = df.iloc[best_idxs]

    best_df.to_csv(path + "best_gcn_runs.csv")
    print(best_df)

if __name__ == "__main__":
    path = "../ablations/"
    name = "rebuttal_gcn_ablation.csv"
    import_csv(path, name)