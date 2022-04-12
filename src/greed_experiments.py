from run_GNN import main
from greed_params import greed_run_params, not_sweep_args, tf_ablation_args, default_params

def run_experiments():
    opt = default_params()
    opt = greed_run_params(opt)
    if opt['function'] in ['greed', 'greed_scaledDP', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero',
                           'greed_non_linear']:
        opt = greed_run_params(opt)

    if not opt['wandb_sweep']:  # sweeps are run from YAML config so don't need these
        opt = not_sweep_args(opt, project_name='greed_runs', group_name='testing')

    # greed_non_linear params
    opt['use_best_params'] = False
    opt['method'] = 'euler'
    opt['step_size'] = 1.0
    opt['epoch'] = 129
    opt['num_splits'] = 1
    opt['optimizer'] = 'adam'
    opt['lr'] = 0.005
    opt['dropout'] = 0.6
    opt['decay'] = 0.0
    opt['gnl_style'] = 'general_graph'
    opt['gnl_savefolder'] = 'reporting_runs'
    opt['gnl_thresholding'] = False


    for data in ['chameleon', 'Cora']:
        if data == 'chameleon':
            opt['geom_gcn_splits'] = True
        elif data == 'Cora':
            opt['geom_gcn_splits'] = False
        opt['dataset'] = data

        for gnl_measure in ['ones', 'deg_poly', 'nodewise', 'nodewise_exp', 'deg_poly_exp']:
            opt['gnl_measure'] = gnl_measure

            for drift in [True, False]:
                opt['drift'] = drift

                for gnl_W_style in ['sum', 'prod', 'k_diag', 'k_block' 'cgnn']:
                    opt['gnl_W_style'] = gnl_W_style

                    for time in [3, 8]:
                        opt['time'] = time

                        for hidden_dim in [64, 512]:
                            opt['hidden_dim'] = hidden_dim

                            opt = tf_ablation_args(opt)
                            main(opt)

if __name__ == "__main_":
    run_experiments()