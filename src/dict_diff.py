import pandas as pd

#original best
d1 = {'use_best_params': True, 'gpu': 0, 'epoch': 200, 'patience': None, 'optimizer': 'adam', 'lr': 0.005823, 'decay': 0.0001821, 'dataset': 'squirrel', 'data_norm': 'rw', 'self_loop_weight': None, 'use_labels': False, 'label_rate': 0.5, 'planetoid_split': False, 'geom_gcn_splits': True, 'num_splits': 1, 'not_lcc': True, 'hetero_SL': True, 'hetero_undir': True, 'block': 'constant', 'function': 'graff', 'hidden_dim': 64, 'fc_out': False, 'input_dropout': 0.5094, 'dropout': 0.4974, 'batch_norm': False, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'use_mlp': False, 'add_source': True, 'XN_activation': False, 'm2_mlp': False, 'time': 2.339, 'augment': False, 'method': 'euler', 'step_size': 1, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'max_nfe': 1000, 'max_test_steps': 100, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'gcn_fixed': 'False', 'gcn_enc_dec': 'False', 'gcn_non_lin': 'False', 'gcn_symm': 'False', 'gcn_bias': 'False', 'gcn_mid_dropout': 'False', 'gcn_params': None, 'gcn_params_idx': 0, 'omega_style': 'zero', 'omega_diag': 'free', 'omega_params': None, 'w_style': 'diag_dom', 'w_diag_init': 'uniform', 'w_param_free': True, 'w_diag_init_q': 1.0, 'w_diag_init_r': 0.0, 'w_params': None, 'time_dep_w': False, 'time_dep_struct_w': False, 'graff_params': ['diag_dom', 'uniform', True, 'diag', 'free', False, True, True], 'omega': 'diag', 'test_mu_0': True, 'device': 'cpu'}
#local best
d2 = {'use_cora_defaults': False, 'dataset': 'squirrel', 'data_feat_norm': True, 'data_norm': 'rw', 'self_loop_weight': 0.0, 'use_labels': False, 'label_rate': 0.5, 'planetoid_split': False, 'geom_gcn_splits': True, 'num_splits': 1, 'hidden_dim': 64, 'fc_out': False, 'input_dropout': 0.5094, 'dropout': 0.4974, 'batch_norm': False, 'optimizer': 'adam', 'lr': 0.005823, 'decay': 0.0001821, 'epoch': 251, 'patience': None, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'block': 'constant', 'function': 'greed_non_linear', 'time': 2.339, 'augment': False, 'method': 'euler', 'step_size': 1, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'ode_blocks': 1, 'max_nfe': 5000, 'earlystopxT': 3, 'max_test_steps': 100, 'leaky_relu_slope': 0.2, 'attention_dropout': 0.0, 'heads': 4, 'attention_norm_idx': 0, 'attention_dim': 64, 'mix_features': False, 'reweight_attention': False, 'attention_type': 'scaled_dot', 'square_plus': False, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'not_lcc': True, 'rewiring': None, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 64, 'gdc_threshold': 0.0001, 'gdc_avg_degree': 64, 'ppr_alpha': 0.05, 'heat_time': 3.0, 'att_samp_pct': 1, 'use_flux': False, 'exact': False, 'M_nodes': 64, 'new_edges': 'random', 'sparsify': 'S_hat', 'threshold_type': 'topk_adj', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'KNN_online': False, 'KNN_online_reps': 4, 'KNN_space': 'pos_distance', 'beltrami': False, 'fa_layer': False, 'pos_enc_type': 'DW64', 'pos_enc_orientation': 'row', 'feat_hidden_dim': 64, 'pos_enc_hidden_dim': 32, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_epoch': 5, 'edge_sampling_add': 0.64, 'edge_sampling_add_type': 'importance', 'edge_sampling_rmv': 0.32, 'edge_sampling_sym': False, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_space': 'attention', 'fa_layer_edge_sampling_rmv': 0.8, 'gpu': 0, 'pos_enc_csv': False, 'pos_dist_quantile': 0.001, 'wandb': True, 'wandb_offline': False, 'wandb_sweep': False, 'wandb_watch_grad': False, 'wandb_track_grad_flow': False, 'wandb_entity': 'graph_neural_diffusion', 'wandb_project': 'greed', 'wandb_group': 'testing', 'wandb_run_name': '0920_134524_wandb', 'wandb_output_dir': './wandb_output', 'wandb_epoch_list': [1, 4, 8, 16, 32, 64, 96, 128], 'run_track_reports': False, 'save_wandb_reports': True, 'save_local_reports': False, 'tau_reg': 2, 'test_mu_0': True, 'test_no_chanel_mix': True, 'test_omit_metric_L': False, 'test_omit_metric_R': False, 'test_tau_remove_tanh': False, 'test_tau_symmetric': False, 'test_tau_outside': True, 'test_linear_L0': True, 'test_R1R2_0': True, 'test_grand_metric': True, 'test_tau_ones': True, 'use_mlp': False, 'add_source': False, 'no_early': True, 'symmetric_QK': False, 'symmetric_attention': True, 'sym_row_max': False, 'use_best_params': True, 'greed_momentum': False, 'momentum_alpha': 0.2, 'dim_p_omega': 16, 'dim_p_w': 16, 'gamma_epsilon': 0.01, 'XN_no_activation': True, 'm2_mlp': False, 'attention_activation': 'exponential', 'attention_normalisation': 'none', 'T0term_normalisation': 'T0_rowSum', 'laplacian_norm': 'lap_symmDeg_RowSumnorm', 'R_T0term_normalisation': 'T0_identity', 'R_laplacian_norm': 'lap_noNorm', 'alpha_style': 'sigmoid', 'fix_alpha': None, 'diffusion': True, 'repulsion': False, 'W_type': 'identity', 'R_W_type': 'identity', 'R_depon_A': 'none', 'W_beta': 0.5, 'tau_residual': True, 'drift': False, 'gnl_thresholding': False, 'lie_trotter': None, 'time2': None, 'time3': None, 'gcn_fixed': False, 'gcn_enc_dec': False, 'gcn_non_lin': False, 'gcn_symm': False, 'gcn_bias': False, 'gcn_mid_dropout': False, 'gcn_params': None, 'gcn_params_idx': 0, 'gnl_style': 'general_graph', 'gnl_activation': 'identity', 'gnl_measure': 'ones', 'gnl_omega': 'diag', 'gnl_omega_diag': 'free', 'gnl_omega_activation': 'identity', 'gnl_W_style': 'diag_dom', 'gnl_attention': False, 'k_blocks': 5, 'block_size': 5, 'k_diags': 11, 'k_diag_pc': 0.1, 'gnl_W_diag_init': 'uniform', 'gnl_W_param_free': True, 'gnl_W_diag_init_q': 1.0, 'gnl_W_diag_init_r': 0.0, 'gnl_W_norm': False, 'two_hops': False, 'time_dep_w': None, 'time_dep_omega': None, 'time_dep_q': None, 'num_lamb_w': 2, 'num_lamb_omega': 2, 'num_lamb_q': 2, 'target_homoph': '0.80', 'hetero_SL': True, 'hetero_undir': True, 'gnl_savefolder': 'animation_runs', 'omega_params': None, 'W_params': None, 'greed_params': None, 'loss_reg': None, 'loss_reg_weight': 1.0, 'loss_reg_delay': 0, 'loss_reg_certainty': None, 'm2_aug': False, 'm1_W_eig': False, 'm2_W_eig': False, 'm3_path_dep': None, 'path_dep_norm': None, 'drift_space': 'label', 'drift_grad': True, 'loss_orthog_a': 0.0, 'householder_L': 8, 'source_term': 'scalar', 'post_proc': False, 'dampen_gamma': 1.0, 'pointwise_nonlin': False, 'conv_batch_norm': False, 'batch': 128, 'graph_pool': '', 'lr_reduce_factor': 0.5, 'lr_schedule_patience': 10, 'lr_schedule_threshold': 0.0001, 'min_lr': 1e-05, 'lt_pointwise_nonlin': False, 'lt_block_times': None, 'lt_block_type': '', 'lt_block_time': None, 'lt_block_step': None, 'lt_block_dimension': 64, 'share_block': '', 'display_epoch_list': [16, 128], 'gnl_omega_diag_val': None, 'reports_list': [1, 4, 12], 'threshold_times': [2, 4], 'w_style': 'diag_dom', 'device': 'cpu'}
#best current sweep
d3 = {'XN_no_activation': False, 'block': 'constant', 'conv_batch_norm': False, 'data_feat_norm': True, 'dataset': 'squirrel', 'decay': 0.0030163897114559393, 'drift': False, 'dropout': 0.27614970270548184, 'epoch': 200, 'function': 'greed_non_linear', 'gnl_W_norm': False, 'gnl_W_style': 'diag_dom', 'gnl_activation': 'identity', 'gnl_measure': 'ones', 'gnl_omega': 'diag', 'gnl_omega_diag': 'free', 'gnl_style': 'general_graph', 'hidden_dim': 32, 'input_dropout': 0.5543434764559019, 'lr': 0.00290650060291862, 'm1_W_eig': False, 'm2_W_eig': 'eye', 'method': 'euler', 'no_early': True, 'num_splits': 10, 'optimizer': 'adam', 'pointwise_nonlin': False, 'self_loop_weight': 0, 'source_term': 'scalar', 'step_size': 1, 'time': 2.0370759858359597, 'time_dep_omega': 'none', 'time_dep_q': 'none', 'time_dep_w': 'none', 'two_hops': True, 'use_cora_defaults': False, 'data_norm': 'rw', 'use_labels': False, 'label_rate': 0.5, 'planetoid_split': False, 'geom_gcn_splits': True, 'fc_out': False, 'batch_norm': False, 'patience': None, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'augment': False, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'ode_blocks': 1, 'max_nfe': 1000, 'earlystopxT': 3, 'max_test_steps': 100, 'leaky_relu_slope': 0.2, 'attention_dropout': 0.0, 'heads': 4, 'attention_norm_idx': 0, 'attention_dim': 64, 'mix_features': False, 'reweight_attention': False, 'attention_type': '', 'square_plus': False, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'not_lcc': True, 'rewiring': None, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 64, 'gdc_threshold': 0.0001, 'gdc_avg_degree': 64, 'ppr_alpha': 0.05, 'heat_time': 3.0, 'att_samp_pct': 1, 'use_flux': False, 'exact': False, 'M_nodes': 64, 'new_edges': 'random', 'sparsify': 'S_hat', 'threshold_type': 'topk_adj', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'KNN_online': False, 'KNN_online_reps': 4, 'KNN_space': 'pos_distance', 'beltrami': False, 'fa_layer': False, 'pos_enc_type': 'DW64', 'pos_enc_orientation': 'row', 'feat_hidden_dim': 64, 'pos_enc_hidden_dim': 32, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_epoch': 5, 'edge_sampling_add': 0.64, 'edge_sampling_add_type': 'importance', 'edge_sampling_rmv': 0.32, 'edge_sampling_sym': False, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_space': 'attention', 'fa_layer_edge_sampling_rmv': 0.8, 'gpu': 0, 'pos_enc_csv': False, 'pos_dist_quantile': 0.001, 'wandb': True, 'wandb_offline': False, 'wandb_sweep': True, 'wandb_watch_grad': False, 'wandb_track_grad_flow': False, 'wandb_entity': 'graph_neural_diffusion', 'wandb_project': 'greed', 'wandb_group': 'testing', 'wandb_run_name': None, 'wandb_output_dir': './wandb_output', 'wandb_epoch_list': [1, 2, 4, 8, 16, 32, 64, 96, 128, 254], 'run_track_reports': False, 'save_wandb_reports': False, 'save_local_reports': False, 'tau_reg': 2, 'test_mu_0': True, 'test_no_chanel_mix': True, 'test_omit_metric_L': True, 'test_omit_metric_R': True, 'test_tau_remove_tanh': True, 'test_tau_symmetric': True, 'test_tau_outside': True, 'test_linear_L0': True, 'test_R1R2_0': True, 'test_grand_metric': True, 'test_tau_ones': True, 'use_mlp': False, 'add_source': False, 'symmetric_QK': False, 'symmetric_attention': False, 'sym_row_max': False, 'use_best_params': False, 'greed_momentum': False, 'momentum_alpha': 0.2, 'dim_p_omega': 16, 'dim_p_w': 16, 'gamma_epsilon': 0.01, 'm2_mlp': False, 'attention_activation': 'exponential', 'attention_normalisation': 'sym_row_col', 'T0term_normalisation': 'T0_identity', 'laplacian_norm': 'lap_noNorm', 'R_T0term_normalisation': 'T0_identity', 'R_laplacian_norm': 'lap_noNorm', 'alpha_style': '', 'fix_alpha': None, 'diffusion': True, 'repulsion': False, 'W_type': 'identity', 'R_W_type': 'identity', 'R_depon_A': '', 'W_beta': 0.5, 'tau_residual': False, 'gnl_thresholding': False, 'lie_trotter': None, 'time2': None, 'time3': None, 'gcn_fixed': False, 'gcn_enc_dec': False, 'gcn_non_lin': False, 'gcn_symm': False, 'gcn_bias': False, 'gcn_mid_dropout': False, 'gcn_params': None, 'gcn_params_idx': 0, 'gnl_omega_activation': 'identity', 'gnl_attention': False, 'k_blocks': 5, 'block_size': 5, 'k_diags': 11, 'k_diag_pc': 0.1, 'gnl_W_diag_init': 'identity', 'gnl_W_param_free': True, 'gnl_W_diag_init_q': 1.0, 'gnl_W_diag_init_r': 0.0, 'num_lamb_w': 1, 'num_lamb_omega': 1, 'num_lamb_q': 1, 'target_homoph': '0.80', 'hetero_SL': True, 'hetero_undir': True, 'gnl_savefolder': '', 'omega_params': None, 'W_params': None, 'greed_params': None, 'loss_reg': None, 'loss_reg_weight': 1.0, 'loss_reg_delay': 0.0, 'loss_reg_certainty': 1.0, 'm2_aug': False, 'm3_path_dep': '', 'path_dep_norm': 'False', 'drift_space': None, 'drift_grad': True, 'loss_orthog_a': 0, 'householder_L': 8, 'post_proc': 'none', 'dampen_gamma': 1.0, 'batch': 128, 'graph_pool': '', 'lr_reduce_factor': 0.5, 'lr_schedule_patience': 10, 'lr_schedule_threshold': 0.0001, 'min_lr': 1e-05, 'lt_pointwise_nonlin': False, 'lt_block_times': None, 'lt_block_type': '', 'lt_block_time': None, 'lt_block_step': None, 'lt_block_dimension': 64, 'share_block': '', 'device': 'cuda'}


# get rid of "TypeError: unhashable type: 'list'"
d1 = {k: tuple(v) if type(v) is list else v for k, v in d1.items()}
d2 = {k: tuple(v) if type(v) is list else v for k, v in d2.items()}
d3 = {k: tuple(v) if type(v) is list else v for k, v in d3.items()}
# d4 = {k: tuple(v) if type(v) is list else v for k, v in d4.items()}

s1 = set(d1.items())
s2 = set(d2.items())
# s3 = set(d3.items())

# print("\n Difference between 71% best and local \n")
# print(s1 ^ s2)
# print("\n Difference between 71% best and current sweep \n")
# print(s3 ^ s1)
# print("\n Difference between current sweep and local \n")
# print(s3 ^ s2)

df1 = pd.DataFrame.from_dict(d1, orient='index')
df2 = pd.DataFrame.from_dict(d2, orient='index')
df3 = pd.DataFrame.from_dict(d3, orient='index')
# df4 = pd.DataFrame.from_dict(d4, orient='index')

print("\n Merged dataframe \n")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# df = pd.concat([df1, df2], axis=1, join='outer')
df = pd.concat([df1, df2, df3], axis=1, join='outer')
# df = pd.concat([df2, df3], axis=1, join='outer')
# df = pd.concat([df2, df3, df4], axis=1, join='outer')

print(df)
df.to_csv("hp_dff.csv")