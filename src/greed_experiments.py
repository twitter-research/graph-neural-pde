import datetime
from run_GNN import main
from greed_params import greed_run_params, not_sweep_args, tf_ablation_args, default_params

def run_experiments():
    opt = default_params()

    #wandb args
    opt['wandb'] = True #False #True
    opt['wandb_track_grad_flow'] = True #False  # don't plot grad flows when testing
    opt['wandb_watch_grad'] = False
    opt['run_track_reports'] = True
    opt['wandb_reports'] = True
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

            for drift in [False, True]:
                opt['drift'] = drift

                for gnl_W_style in ['sum', 'prod', 'k_diag', 'k_block' 'cgnn']:
                    opt['gnl_W_style'] = gnl_W_style

                    for time in [3, 8]:
                        opt['time'] = time

                        for hidden_dim in [64, 512]:
                            opt['hidden_dim'] = hidden_dim

                            opt = tf_ablation_args(opt)
                            opt['gnl_savefolder'] = f"{data}_{gnl_measure}_drift{str(drift)}_W{gnl_W_style}_t{str(time)}_hd{str(hidden_dim)}"

                            main(opt)

if __name__ == "__main__":
    run_experiments()