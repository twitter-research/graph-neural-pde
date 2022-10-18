# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

best_params_dict_L = {
'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0.4327 ,'hidden_dim': 64 ,'time': 3.194 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},                       
# 'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0 ,'hidden_dim': 2325,'time': 3 ,'step_size': 1,
                    #    'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag",  "scalable": True},
'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.0027 ,'decay': 0.0006 ,'dropout': 0.159 ,'input_dropout': 0.349 ,'hidden_dim': 128 ,'time': 3.275 ,'step_size': 1 ,
              'conv_batch_norm': "layerwise", "source_term": "none", "omega_style": "zero"},
# 'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.0027 ,'decay': 0.0006 ,'dropout': 0.159 ,'input_dropout': 0,'hidden_dim': 2089 ,'time': 3 ,'step_size': 1 ,
#               'conv_batch_norm': "none", "source_term": "none", "omega_style": "zero",  "scalable": True},
'texas': { 'w_style': 'diag_dom' ,'lr': 0.004145 ,'decay': 0.03537 ,'dropout': 0.3293 ,'input_dropout': 0.3936 ,'hidden_dim': 64 ,'time': 0.5756 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
'wisconsin': { 'w_style': 'diag' ,'lr': 0.002908 ,'decay': 0.03178 ,'dropout': 0.3717 ,'input_dropout': 0.3674 ,'hidden_dim': 64 ,'time': 2.099 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
'cornell': { 'w_style': 'diag' ,'lr': 0.002105 ,'decay': 0.01838 ,'dropout': 0.2978 ,'input_dropout': 0.4421 ,'hidden_dim': 64 ,'time': 2.008 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
'film': { 'w_style': 'diag' ,'lr': 0.002602 ,'decay': 0.01299 ,'dropout': 0.4847 ,'input_dropout': 0.4191 ,'hidden_dim': 64 ,'time': 1.541 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
# 'Cora': { 'w_style': 'diag' ,'lr': 0.00261 ,'decay': 0.04125 ,'dropout': 0.3386 ,'input_dropout': 0.5294 ,'hidden_dim': 64 ,'time': 3 ,'step_size': 0.25 ,
#                        'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
'Cora': { 'w_style': 'diag' ,'lr': 0.00261 ,'decay': 0.04125 ,'dropout': 0.3386 ,'input_dropout': 0 ,'hidden_dim': 1433 ,'time': 3 ,'step_size': 1 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag",  "scalable": True},
'Citeseer': { 'w_style': 'diag' ,'lr': 0.000117 ,'decay': 0.02737 ,'dropout': 0.2224 ,'input_dropout': 0.5129 ,'hidden_dim': 64 ,'time': 2 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
# 'Citeseer': { 'w_style': 'diag' ,'lr': 0.000117 ,'decay': 0.02737 ,'dropout': 0.2224 ,'input_dropout': 0 ,'hidden_dim': 3703 ,'time': 2 ,'step_size': 1 ,
#                        'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag",  "scalable": True},
'Pubmed': { 'w_style': 'diag' ,'lr': 0.00394 ,'decay': 0.0003348 ,'dropout': 0.4232 ,'input_dropout': 0.412 ,'hidden_dim': 64 ,'time': 2.552 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag"},
# 'Pubmed': { 'w_style': 'diag' ,'lr': 0.00394 ,'decay': 0.0003348 ,'dropout': 0.4232 ,'input_dropout': 0 ,'hidden_dim': 500 ,'time': 2 ,'step_size': 1,
#                        'conv_batch_norm': "none", "source_term": "scalar", "omega_style": "diag",  "scalable": True},                       
                       
}
                       

best_params_dict_NL = {
'chameleon': { 'w_style': 'diag_dom' , 'lr': 0.0050 ,'decay': 0.0005 ,'dropout': 0.3577 ,'input_dropout': 0.4756 ,'hidden_dim': 64 ,'time': 3.331 ,'step_size': 0.5,
                       'conv_batch_norm': "layerwise", "source_term": "none", "gnl_omega": "zero"},
'squirrel': { 'w_style': 'diag_dom' , 'lr': 0.0065 ,'decay': 0.0009 ,'dropout': 0.1711 ,'input_dropout': 0.3545 ,'hidden_dim': 128 ,'time': 2.871 ,'step_size': 0.5,
              'conv_batch_norm': "layerwise", "source_term": "none", "gnl_omega": "zero"},
'texas': { 'w_style': 'diag_dom' ,'lr': 0.0042 ,'decay': 0.0175 ,'dropout': 0.2346 ,'input_dropout': 0.4037 ,'hidden_dim': 32 ,'time': 2.656 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'wisconsin': { 'w_style': 'diag_dom' , 'lr': 0.0043 ,'decay': 0.0345 ,'dropout': 0.3575 ,'input_dropout': 0.3508 ,'hidden_dim': 32 ,'time': 3.785 ,'step_size': 1 ,
                       'conv_batch_norm': "layerwise", "source_term": "scalar", "gnl_omega": "diag"},
'cornell': { 'w_style': 'diag' , 'lr': 0.0049 ,'decay': 0.0431 ,'dropout': 0.3576 ,'input_dropout': 0.4365 ,'hidden_dim': 32 ,'time': 2.336 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'film': { 'w_style': 'diag_dom' , 'lr': 0.0049 ,'decay': 0.0163 ,'dropout': 0.3682 ,'input_dropout': 0.4223 ,'hidden_dim': 32 ,'time': 1.114 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'Cora': { 'w_style': 'diag_dom' , 'lr': 0.0030 ,'decay': 0.0263 ,'dropout': 0.4241 ,'input_dropout': 0.5378 ,'hidden_dim': 64 ,'time': 1.445 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'Citeseer': { 'w_style': 'diag' , 'lr': 0.0016 ,'decay': 0.0065 ,'dropout': 0.3846 ,'input_dropout': 0.4389 ,'hidden_dim': 64 ,'time': 2.136 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'Pubmed': { 'w_style': 'diag' , 'lr': 0.0048 ,'decay': 0.0002 ,'dropout': 0.5292 ,'input_dropout': 0.414 ,'hidden_dim': 64 ,'time': 3.343 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}}

def shared_graff_params(opt):
    opt['block'] = 'constant'
    opt['function'] = 'graff'
    opt['optimizer'] = 'adam'
    opt['epoch'] = 250
    opt['method'] = 'euler'
    opt['geom_gcn_splits'] = True
    opt['fc_out'] = True
    return opt

def hetero_params(opt):
    #added self loops and make undirected for chameleon & squirrel
    if opt['dataset'] in ['chameleon', 'squirrel']:
        opt['hetero_SL'] = True
        opt['hetero_undir'] = True
    return opt

# def t_or_f(tf_str):
#     if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
#         return True
#     else:
#         return False

# def tf_ablation_args(opt):
#     tf_args = ['test_no_chanel_mix','test_omit_metric_L', 'test_omit_metric_R','test_mu_0',
#                 'test_tau_remove_tanh','test_tau_symmetric','test_grand_metric','test_tau_ones',
#                 'test_tau_outside', 'test_linear_L0', 'test_R1R2_0',
#                 'use_mlp', 'no_early', 'use_labels', 'hetero_SL', 'hetero_undir',
#                 'add_source', 'symmetric_attention', 'sym_row_max','symmetric_QK',
#                 'diffusion', 'repulsion', 'drift', 'tau_residual',
#                 'm2_mlp', 'gnl_thresholding', 'gnl_W_param_free', 'gnl_W_param_free2', 'gnl_attention',
#                 'XN_no_activation', 'two_hops',
#                 'greed_SL', 'greed_undir', 'm2_aug', 'm1_W_eig', 'gnl_W_norm', 'drift_grad',
#                 'pointwise_nonlin', 'lt_pointwise_nonlin', 'data_feat_norm', 'dir_grad_flow',
#                 'gcn_enc_dec', 'gcn_fixed', 'gcn_non_lin', 'gcn_symm', 'gcn_bias', 'gcn_mid_dropout',
#                 'wandb', 'wandb_sweep', 'wandb_offline']#, 'adjoint']
#     arg_intersect = list(set(opt.keys()) & set(tf_args))
#     for arg in arg_intersect:
#         str_tf = opt[arg]
#         bool_tf = t_or_f(str_tf)
#         opt[arg] = bool_tf

#     return opt