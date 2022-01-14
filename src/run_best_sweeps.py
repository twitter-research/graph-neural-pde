import yaml
from greed_params import default_params, not_sweep_args, greed_run_params
from run_GNN import main

def run_best(sweep, run_list, project_name, group_name, num_runs):
    for run in run_list:
        default_params_dict = default_params()
        greed_run_dict = greed_run_params(opt)
        not_sweep_dict = not_sweep_args(default_params_dict, project_name, group_name)

        yaml_path = f"./wandb/sweep-{sweep}/config-{run}.yaml"
        with open(yaml_path) as f:
            yaml_opt = yaml.load(f, Loader=yaml.FullLoader)
        temp_opt = {}
        for k, v in yaml_opt.items():
            if type(v) == dict:
                temp_opt[k] = v['value']
            else:
                temp_opt[k] = v
        yaml_opt = temp_opt

        opt = {**default_params_dict, **greed_run_dict, **not_sweep_dict, **yaml_opt}
        opt['wandb_best_run_id'] = run

        for i in range(num_runs):
            main(opt)

if __name__ == "__main__":
    sweep = 'ebq1b5hy'
    run_list = ['yv3v42ym', '7ba0jl9m', 'a60dnqcc', 'v6ln1x90', 'f5dmv6ow']
    project_name = 'best_runs'
    group_name = 'eval'
    run_best(sweep, run_list, project_name, group_name, num_runs=8)