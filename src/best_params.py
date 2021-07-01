import argparse

best_params_dict = {'Cora': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 128, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': True, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Cora', 'decay': 0.00507685443154266, 'directional_penalty': None, 'dropout': 0.046878964627763316, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': True, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 0.5, 'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.5, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.022924849756740397, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 2000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'cora_beltrami_splits', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 1000, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adamax', 'patience': 100, 'pos_enc_hidden_dim': 16, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 18.294754260552843, 'tol_scale': 821.9773048827274, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},
                    'Citeseer': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 32, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'exp_kernel', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': True, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Citeseer', 'decay': 0.1, 'directional_penalty': None, 'dropout': 0.7488085003122172, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 250, 'exact': True, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 128, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.6803233752085334, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.5825086997804176, 'lr': 0.00863585231323069, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 3000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'Citeseer_beltrami_1_KNN', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_class': 6, 'num_feature': 3703, 'num_init': 2, 'num_nodes': 2120, 'num_samples': 400, 'num_splits': 0, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 4, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 7.874113442879092, 'tol_scale': 2.9010446330432815, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},
                    'Pubmed': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': True, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 16, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'cosine_sim', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': True, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Pubmed', 'decay': 0.0018236722171703636, 'directional_penalty': None, 'dropout': 0.07191100715473969, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 600, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 1, 'heat_time': 3.0, 'hidden_dim': 128, 'input_dropout': 0.5, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.014669345840305131, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 5000, 'method': 'dopri5', 'metric': 'test_acc', 'mix_features': False, 'name': None, 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 8, 'ode_blocks': 1, 'optimizer': 'adamax', 'patience': 100, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 12.942327880200853, 'tol_scale': 1991.0688305523001, 'tol_scale_adjoint': 16324.368093998313, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False, 'folder': 'pubmed_linear_att_beltrami_adj2', 'index': 0, 'run_with_KNN': False, 'change_att_sim_type': False, 'reps': 1, 'max_test_steps': 100, 'no_early': False, 'earlystopxT': 5.0, 'pos_enc_csv': False, 'pos_enc_type': 'GDC'},
                    'CoauthorCS': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'dopri5', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 8, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': True, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'CoauthorCS', 'decay': 0.004738413087298854, 'directional_penalty': None, 'dropout': 0.6857774850321, 'dt': 0.001, 'dt_min': 1e-05, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_add': 0.05, 'edge_sampling_epoch': 5, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_rmv': 0.05, 'edge_sampling_space': 'pos_distance', 'edge_sampling_sym': False, 'epoch': 250, 'exact': False, 'fa_layer': False, 'fc_out': False, 'feat_hidden_dim': 128, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.0001, 'gpus': 1, 'grace_period': 20, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 16, 'input_dropout': 0.5275042493231822, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.7181389780997276, 'lr': 0.0009342860080741642, 'max_iters': 100, 'max_nfe': 3000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'CoauthorCS_final_tune_posencGDC', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 4, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'pos_dist_quantile': 0.001, 'pos_enc_csv': False, 'pos_enc_hidden_dim': 32, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 0, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'symmetric_attention': False, 'threshold_type': 'addD_rvR', 'time': 3.126400580172773, 'tol_scale': 9348.983916372074, 'tol_scale_adjoint': 6599.1250595331385, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'Computers': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'dopri5', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.572918052062338, 'attention_dim': 64, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': True, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Computers', 'decay': 0.007674669913252157, 'directional_penalty': None, 'dropout': 0.08732611854459256, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 25, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 128, 'input_dropout': 0.5973137276937647, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.0035304663972281548, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'computer_beltrami_hard_att1', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_hidden_dim': 32, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW128', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1.7138583550928912, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.249016177876166, 'tol_scale': 127.46369887079446, 'tol_scale_adjoint': 443.81436775321754, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'Photo': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.9282359956104751, 'attention_dim': 64, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'pearson', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': True, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Photo', 'decay': 0.004707800883497945, 'directional_penalty': None, 'dropout': 0.46502284638600183, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 25, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 64, 'input_dropout': 0.42903126506740247, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.005560726683883279, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'photo_beltrami_hard_att1', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_hidden_dim': 16, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW128', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 0.05783612585280118, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.5824027975386623, 'tol_scale': 2086.525473167121, 'tol_scale_adjoint': 14777.606112557354, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'ogbn-arxiv': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.8105268910037231, 'attention_dim': 32, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': True, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'ogbn-arxiv', 'decay': 0, 'directional_penalty': None, 'dropout': 0.11594990901233933, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 2, 'heat_time': 3.0, 'hidden_dim': 162, 'input_dropout': 0, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.21964773835397075, 'leaky_relu_slope': 0.2, 'lr': 0.005451476553977102, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'arxiv_beltrami_hard_att', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': False, 'num_init': 2, 'num_samples': 200, 'num_splits': 0, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'patience': 100, 'pos_enc_hidden_dim': 98, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW64', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.6760155951687636, 'tol_scale': 11353.558848254957, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False}
                    }

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

args = parser.parse_args(args=[])

default_args = vars(args)