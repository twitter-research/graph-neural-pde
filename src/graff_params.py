import argparse
import  wandb
import pandas as pd

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
#only squirrel updated # the rest keep Neurips defaults for 'conv_batch_norm'/"source_term"/"gnl_omega". Gives scope to improve others numbers.
best_params_dict_lin = {
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

# https://wandb.ai/graph_neural_diffusion/penn94/reports/ICLR-experiments--VmlldzoyNzEwNTUw/edit
best_params_dict_NL = {
'chameleon': { 'w_style': 'diag_dom' , 'lr': 0.0050 ,'decay': 0.0005 ,'dropout': 0.3577 ,'input_dropout': 0.4756 ,'hidden_dim': 64 ,'time': 3.331 ,'step_size': 0.5,
                       'conv_batch_norm': "layerwise", "source_term": "none", "gnl_omega": "zero"}, #71.38 \pm 1.47
'squirrel': { 'w_style': 'diag_dom' , 'lr': 0.0065 ,'decay': 0.0009 ,'dropout': 0.1711 ,'input_dropout': 0.3545 ,'hidden_dim': 128 ,'time': 2.871 ,'step_size': 0.5,
              'conv_batch_norm': "layerwise", "source_term": "none", "gnl_omega": "zero"}, #59.01 \pm 1.31
'texas': { 'w_style': 'diag_dom' ,'lr': 0.0042 ,'decay': 0.0175 ,'dropout': 0.2346 ,'input_dropout': 0.4037 ,'hidden_dim': 32 ,'time': 2.656 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}, #86.49 \pm 4.84
'wisconsin': { 'w_style': 'diag_dom' , 'lr': 0.0043 ,'decay': 0.0345 ,'dropout': 0.3575 ,'input_dropout': 0.3508 ,'hidden_dim': 32 ,'time': 3.785 ,'step_size': 1 ,
                       'conv_batch_norm': "layerwise", "source_term": "scalar", "gnl_omega": "diag"}, #87.26 \pm 2.52
'cornell': { 'w_style': 'diag' , 'lr': 0.0049 ,'decay': 0.0431 ,'dropout': 0.3576 ,'input_dropout': 0.4365 ,'hidden_dim': 32 ,'time': 2.336 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}, #77.30 \pm 3.24
'film': { 'w_style': 'diag_dom' , 'lr': 0.0049 ,'decay': 0.0163 ,'dropout': 0.3682 ,'input_dropout': 0.4223 ,'hidden_dim': 32 ,'time': 1.114 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}, #35.96 \pm 0.95
'Cora': { 'w_style': 'diag_dom' , 'lr': 0.0030 ,'decay': 0.0263 ,'dropout': 0.4241 ,'input_dropout': 0.5378 ,'hidden_dim': 64 ,'time': 1.445 ,'step_size': 1,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}, #87.81 \pm 1.13
'Citeseer': { 'w_style': 'diag' , 'lr': 0.0016 ,'decay': 0.0065 ,'dropout': 0.3846 ,'input_dropout': 0.4389 ,'hidden_dim': 64 ,'time': 2.136 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}, #76.81 \pm 1.12
'Pubmed': { 'w_style': 'diag' , 'lr': 0.0048 ,'decay': 0.0002 ,'dropout': 0.5292 ,'input_dropout': 0.414 ,'hidden_dim': 64 ,'time': 3.343 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "source_term": "scalar", "gnl_omega": "diag"}} #89.81 \pm 0.50


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

df = pd.DataFrame.from_dict(best_params_dict_lin)
cols = ["w_style", "source_term", "gnl_omega","conv_batch_norm"]
print(df.transpose()[cols])
df = pd.DataFrame.from_dict(best_params_dict_NL)
print(df.transpose()[cols])