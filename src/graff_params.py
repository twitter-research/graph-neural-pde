# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

best_params_dict_L = {
'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0.4327 ,'hidden_dim': 64 ,'time': 3.194 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.0027 ,'decay': 0.0006 ,'dropout': 0.159 ,'input_dropout': 0.349 ,'hidden_dim': 128 ,'time': 3.275 ,'step_size': 1 ,
              'conv_batch_norm': "layerwise", "source_term": "none", "gnl_omega": "zero"},
'texas': { 'w_style': 'diag_dom' ,'lr': 0.004145 ,'decay': 0.03537 ,'dropout': 0.3293 ,'input_dropout': 0.3936 ,'hidden_dim': 64 ,'time': 0.5756 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'wisconsin': { 'w_style': 'diag' ,'lr': 0.002908 ,'decay': 0.03178 ,'dropout': 0.3717 ,'input_dropout': 0.3674 ,'hidden_dim': 64 ,'time': 2.099 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'cornell': { 'w_style': 'diag' ,'lr': 0.002105 ,'decay': 0.01838 ,'dropout': 0.2978 ,'input_dropout': 0.4421 ,'hidden_dim': 64 ,'time': 2.008 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'film': { 'w_style': 'diag' ,'lr': 0.002602 ,'decay': 0.01299 ,'dropout': 0.4847 ,'input_dropout': 0.4191 ,'hidden_dim': 64 ,'time': 1.541 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'Cora': { 'w_style': 'diag' ,'lr': 0.00261 ,'decay': 0.04125 ,'dropout': 0.3386 ,'input_dropout': 0.5294 ,'hidden_dim': 64 ,'time': 3 ,'step_size': 0.25 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'Citeseer': { 'w_style': 'diag' ,'lr': 0.000117 ,'decay': 0.02737 ,'dropout': 0.2224 ,'input_dropout': 0.5129 ,'hidden_dim': 64 ,'time': 2 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"},
'Pubmed': { 'w_style': 'diag' ,'lr': 0.00394 ,'decay': 0.0003348 ,'dropout': 0.4232 ,'input_dropout': 0.412 ,'hidden_dim': 64 ,'time': 2.552 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}}

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
    return opt

def hetero_params(opt):
    #added self loops and make undirected for chameleon & squirrel
    if opt['dataset'] in ['chameleon', 'squirrel']:
        opt['hetero_SL'] = True
        opt['hetero_undir'] = True
    return opt