"""
Store the global parameter dictionary to be imported and modified by each test
"""

OPT = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
       'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
       'hidden_dim': 6, 'block': 'attention', 'function': 'laplacian', 'augment': False, 'adjoint': False,
       'tol_scale': 1, 'time': 1, 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler', 'rewiring': None,
       'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None,
       'total_deriv': None, 'directional_penalty': None, 'step_size': 1, 'beltrami': False, 'use_mlp': False,
       'use_labels': False, 'fc_out': False, 'attention_type': "scaled_dot", 'batch_norm': False, 'square_plus': False,
       'feat_hidden_dim': 16, 'pos_enc_hidden_dim': 8, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 4,
       'gdc_threshold': 1e-5, 'ppr_alpha': 0.05, 'exact': True, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC',
       'max_nfe': 1000, 'pos_enc_csv': False, 'max_test_steps': 1000, 'edge_sampling_add_type': 'importance',
       'fa_layer': False, 'att_samp_pct': 1, 'edge_sampling_sym': False, 'data_norm': 'rw', 'lr': 0.01, 'decay': 0,
       'max_iters': 1000, 'geom_gcn_splits': False}
