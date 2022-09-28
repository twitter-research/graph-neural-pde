import argparse
import  wandb

#neurips submission
# best_params_dict = {
# 'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0.4327 ,'hidden_dim': 64 ,'time': 3.194 ,'step_size': 1 },
# 'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.005823 ,'decay': 0.0001821 ,'dropout': 0.4974 ,'input_dropout': 0.5094 ,'hidden_dim': 64 ,'time': 2.339 ,'step_size': 1 },
# 'texas': { 'w_style': 'diag_dom' ,'lr': 0.004145 ,'decay': 0.03537 ,'dropout': 0.3293 ,'input_dropout': 0.3936 ,'hidden_dim': 64 ,'time': 0.5756 ,'step_size': 0.5 },
# 'wisconsin': { 'w_style': 'diag' ,'lr': 0.002908 ,'decay': 0.03178 ,'dropout': 0.3717 ,'input_dropout': 0.3674 ,'hidden_dim': 64 ,'time': 2.099 ,'step_size': 0.5 },
# 'cornell': { 'w_style': 'diag' ,'lr': 0.002105 ,'decay': 0.01838 ,'dropout': 0.2978 ,'input_dropout': 0.4421 ,'hidden_dim': 64 ,'time': 2.008 ,'step_size': 1 },
# 'film': { 'w_style': 'diag' ,'lr': 0.002602 ,'decay': 0.01299 ,'dropout': 0.4847 ,'input_dropout': 0.4191 ,'hidden_dim': 64 ,'time': 1.541 ,'step_size': 1 },
# 'Cora': { 'w_style': 'diag' ,'lr': 0.00261 ,'decay': 0.04125 ,'dropout': 0.3386 ,'input_dropout': 0.5294 ,'hidden_dim': 64 ,'time': 3 ,'step_size': 0.25 },
# 'Citeseer': { 'w_style': 'diag' ,'lr': 0.000117 ,'decay': 0.02737 ,'dropout': 0.2224 ,'input_dropout': 0.5129 ,'hidden_dim': 64 ,'time': 2 ,'step_size': 0.5 },
# 'Pubmed': { 'w_style': 'diag' ,'lr': 0.00394 ,'decay': 0.0003348 ,'dropout': 0.4232 ,'input_dropout': 0.412 ,'hidden_dim': 64 ,'time': 2.552 ,'step_size': 0.5 }}
#ICLR
best_params_dict = {
'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0.4327 ,'hidden_dim': 64 ,'time': 3.194 ,'step_size': 1 },
# 'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.005823 ,'decay': 0.0001821 ,'dropout': 0.4974 ,'input_dropout': 0.5094 ,'hidden_dim': 64 ,'time': 2.339 ,'step_size': 1 },
'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.0027 ,'decay': 0.0006 ,'dropout': 0.159 ,'input_dropout': 0.349 ,'hidden_dim': 128 ,'time': 3.275 ,'step_size': 1 ,
              'conv_batch_norm': "layerwise"},
'texas': { 'w_style': 'diag_dom' ,'lr': 0.004145 ,'decay': 0.03537 ,'dropout': 0.3293 ,'input_dropout': 0.3936 ,'hidden_dim': 64 ,'time': 0.5756 ,'step_size': 0.5 },
'wisconsin': { 'w_style': 'diag' ,'lr': 0.002908 ,'decay': 0.03178 ,'dropout': 0.3717 ,'input_dropout': 0.3674 ,'hidden_dim': 64 ,'time': 2.099 ,'step_size': 0.5 },
'cornell': { 'w_style': 'diag' ,'lr': 0.002105 ,'decay': 0.01838 ,'dropout': 0.2978 ,'input_dropout': 0.4421 ,'hidden_dim': 64 ,'time': 2.008 ,'step_size': 1 },
'film': { 'w_style': 'diag' ,'lr': 0.002602 ,'decay': 0.01299 ,'dropout': 0.4847 ,'input_dropout': 0.4191 ,'hidden_dim': 64 ,'time': 1.541 ,'step_size': 1 },
'Cora': { 'w_style': 'diag' ,'lr': 0.00261 ,'decay': 0.04125 ,'dropout': 0.3386 ,'input_dropout': 0.5294 ,'hidden_dim': 64 ,'time': 3 ,'step_size': 0.25 },
'Citeseer': { 'w_style': 'diag' ,'lr': 0.000117 ,'decay': 0.02737 ,'dropout': 0.2224 ,'input_dropout': 0.5129 ,'hidden_dim': 64 ,'time': 2 ,'step_size': 0.5 },
'Pubmed': { 'w_style': 'diag' ,'lr': 0.00394 ,'decay': 0.0003348 ,'dropout': 0.4232 ,'input_dropout': 0.412 ,'hidden_dim': 64 ,'time': 2.552 ,'step_size': 0.5 }}


def shared_graff_params(opt):
    opt['block'] = 'constant'
    opt['function'] = 'graff'
    if opt['w_style'] == 'diag_dom':
       opt['graff_params'] = ['diag_dom', 'uniform', True, 'diag', 'free', False, True, True]
    elif opt['w_style'] == 'diag':
        opt['graff_params'] = ['diag', 'uniform', False, 'diag', 'free', True, True, True]
    opt['optimizer'] = 'adam'
    opt['epoch'] = 200
    opt['method'] = 'euler'
    opt['geom_gcn_splits'] = True
    return opt

def hetero_params(opt):
    # adding self loops and make undirected for chameleon & squirrel
    if opt['dataset'] in ['chameleon', 'squirrel']:
        wandb.config.update({'hetero_SL': True}, allow_val_change=True)
        wandb.config.update({'hetero_undir': True}, allow_val_change=True)
        wandb.config.update({'geom_gcn_splits': True}, allow_val_change=True)

        # if opt['wandb']:
        #     wandb.config.update({'hetero_SL': True}, allow_val_change=True)
        #     wandb.config.update({'hetero_undir': True}, allow_val_change=True)
        #     wandb.config.update({'geom_gcn_splits': True}, allow_val_change=True)
        # else:
        #     opt['hetero_SL'] = True
        #     opt['hetero_undir'] = True
        #     opt['geom_gcn_splits'] = True
    return opt