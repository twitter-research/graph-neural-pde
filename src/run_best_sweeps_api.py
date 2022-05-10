import yaml
import argparse
import pandas as pd
import wandb
from greed_params import default_params, not_sweep_args, greed_run_params
from run_GNN import main as run_GNN_main

#using https://docs.wandb.ai/guides/track/public-api-guide

def load_sweep_ids(series, datasets=[]):
    df = pd.read_csv("../wandb_sweep_configs/_greed_sweeps.csv")
    df_filtered = df[(df.series==series) & (df.dataset.isin(datasets))]
    return df_filtered

def get_top_n(entity, sweeps_df, n=5):
    api = wandb.Api()

    for index, row in sweeps_df.iterrows():
        sweep_id = row.sweeps
        if isinstance(sweep_id, list):
            print("your object is a list !")
        else:
            sweep = api.sweep(f"{entity}/{row.project}/{sweep_id}")
            runs = sorted(sweep.runs,
                          key=lambda run: run.summary.get("val_acc", 0), reverse=True)
            run_configs = []
            # for run in runs:
            for i in range(n):
                run_configs.append(runs[i].config)
    return run_configs


def rerun_runs(run_configs, rerun_dict={}):
    '''rerun with new params in rerun_dict'''
    df_params = default_params()
    for run_config in run_configs:
        opt = {**df_params, **run_config}
        for key, vals in rerun_dict.items():
            for val in vals:
                opt[key] = val
                run_GNN_main(opt)
    #todo specify a project name where the new runs need to go
    #todo find a way to parallelize this over gpus, maybe via generating sweep

if __name__ == "__main__":
    datasets = ['cora', 'citeseer']#, 'pubmed', 'chameleon', 'squirrel', 'actor']
    df_filtered = load_sweep_ids(series="final2", datasets=datasets)
    run_configs = get_top_n(entity="graph_neural_diffusion", sweeps_df=df_filtered, n=5)
    rerun_dict = {'epoch': [2,3]}
    rerun_runs(run_configs, rerun_dict=rerun_dict)