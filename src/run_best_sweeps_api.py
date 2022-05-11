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


def rerun_runs(run_configs, project, rerun_dict={}):
    '''rerun with new params in rerun_dict'''
    df_params = default_params()

    for run_config in run_configs:
        opt = {**df_params, **run_config}
        opt['wandb_project'] = project
        opt['wandb_group'] = project
        for key, vals in rerun_dict.items():
            for val in vals:
                opt[key] = val
                run_GNN_main(opt)
    #todo find a way to parallelize this over gpus, maybe via generating sweep

def rerun_sweep(run_configs, rerun_dict, entity, project):
    '''rerun using sweep over new params in rerun_dict'''
    df_params = default_params()

    for run_config in run_configs:
        opt = {**df_params, **run_config}
        opt['wandb_project'] = project
        opt['wandb_group'] = project

        sweep_params = {key: {"values": value} for key, value in opt.items()}
        for key, val_list, in rerun_dict.items:
            sweep_params[key] = {"values": val_list}
        sweep_params['wandb'] = True
        sweep_params['wandb_sweep'] = True

        sweep_configuration = {
            "program": "run_GNN.py",
            "method": "grid",
            "parameters": sweep_params}

        sweep_id = wandb.sweep(sweep=sweep_configuration, entity=entity, project=project)
        wandb.agent(sweep_id)

if __name__ == "__main__":
    datasets = ['chameleon', 'squirrel', 'actor', 'cora', 'citeseer', 'pubmed', 'squirrel']
    df_filtered = load_sweep_ids(series="final2", datasets=datasets)
    n = 5
    entity = "graph_neural_diffusion"
    project = "final2_reruns"
    run_configs = get_top_n(entity="graph_neural_diffusion", sweeps_df=df_filtered, n=n)
    rerun_dict = {'hidden_dim': [32, 64, 128, 256, 512]}
    # rerun_runs(run_configs, project=project, rerun_dict=rerun_dict)
    rerun_sweep(run_configs, rerun_dict=rerun_dict, entity=entity, project=project)