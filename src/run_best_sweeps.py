import yaml
import argparse

from greed_params import default_params, not_sweep_args, greed_run_params
from run_GNN import main

def run_best(cmd_opt, sweep, run_list, project_name, group_name, num_runs):
    if cmd_opt['run_id']:
        run_list = [cmd_opt['run_id']]
    if cmd_opt['sweep_id']:
        sweep = [cmd_opt['sweep_id']]

    for run in run_list:
        default_params_dict = default_params()
        greed_run_dict = greed_run_params(default_params_dict)
        not_sweep_dict = not_sweep_args(greed_run_dict, project_name, group_name)

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

        opt = {**default_params_dict, **greed_run_dict, **not_sweep_dict, **yaml_opt, **cmd_opt}
        opt['wandb_best_run_id'] = run
        opt['use_best_params'] = False
        for i in range(num_runs):
            main(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau_reg', type=int, default=2)
    parser.add_argument('--test_mu_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_no_chanel_mix', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_remove_tanh', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_symmetric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_outside', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_linear_L0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_R1R2_0', type=str, default='True')  # action='store_true')

    parser.add_argument('--sweep_id', type=str, default='', help="sweep_id for 1 best run")  # action='store_true')
    parser.add_argument('--run_id', type=str, default='', help="run_id for 1 best run")  # action='store_true')

    args = parser.parse_args()
    cmd_opt = vars(args)

    sweep = 'ebq1b5hy'
    run_list = ['yv3v42ym', '7ba0jl9m', 'a60dnqcc', 'v6ln1x90', 'f5dmv6ow']
    project_name = 'best_runs'
    group_name = 'eval'
    run_best(cmd_opt, sweep, run_list, project_name, group_name, num_runs=8)