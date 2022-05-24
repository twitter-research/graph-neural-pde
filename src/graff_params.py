import argparse
import datetime

def graff_opt(opt):
    #GNN params
    opt['add_source'] = True #False #True
    opt['use_mlp'] = False #True
    opt['XN_no_activation'] = True #False
    opt['m2_mlp'] = False #False

    #data params
    # opt['planetoid_split'] = True
    opt['geom_gcn_splits'] = True#True #False#True
    opt['epoch'] = 100
    opt['num_splits'] = 1#10#4#1
    opt['patience'] = None
    opt['method'] = 'euler'

    #run params
    opt['optimizer'] = 'adam'
    opt['lr'] = 0.005
    opt['dropout'] = 0.6
    opt["input_dropout"] = 0.5
    opt['decay'] = 0.0
    opt['hidden_dim'] = 64 #512
    opt['time'] = 3.
    opt['step_size'] = 1.
    opt['use_best_params'] = False#True #False #True
    opt["use_mlp"] = False

    # Omega params
    opt['omega_style'] = 'diag'#'zero'
    if opt['omega_style'] == 'diag':
        opt['omega_diag'] = 'free' #'free 'const'
        if opt['omega_diag'] == 'const':
            opt['omega_diag_val'] = 1

    opt['gnl_omega_diag'] = 'free' #'free 'const'
    opt['gnl_omega_diag_val'] = None #1 #-1 # 1
    opt['gnl_omega_activation'] = 'identity' #identity
    opt['gnl_omega_params'] = ["diag","free","None","identity"] #[opt['gnl_omega'], opt['gnl_omega_diag'], opt['gnl_omega_diag_val'], opt['gnl_omega_activation']]

    # W params
    opt['w_style'] = 'diag_dom' #'diag'

    if opt['w_style'] in ['diag', 'diag_dom']:
        opt['w_diag_init'] = 'uniform'#'identity'
        opt['w_param_free'] = True #'True'
        opt['w_diag_init_q'] = 1.0
        opt['w_diag_init_r'] = 0.0


    opt['time_dep_w'] = False
    opt['time_dep_struct_w'] = False#True

    #gcn params
    # opt['function'] = 'gcn_dgl'#'gcn_res_dgl' #'gcn_dgl'#'greed_non_linear' #'gcn' #'greed_non_linear' #'greed_linear_hetero'
    opt['gcn_enc_dec'] = False #False #True
    opt['gcn_fixed'] = False #False #True
    opt['gcn_symm'] = False#True
    opt['gcn_non_lin'] = True #False #True
    opt['gcn_bias'] = True
    opt['gcn_mid_dropout'] = False
    # opt['gcn_params'] = [0, 'gcn_dgl', False, False, False, True]

    return opt


def t_or_f(tf_str):
    if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
        return True
    else:
        return False

def tf_ablation_args(opt):
    tf_args = ['test_no_chanel_mix','test_omit_metric_L', 'test_omit_metric_R','test_mu_0',
                'test_tau_remove_tanh','test_tau_symmetric','test_grand_metric','test_tau_ones',
                'test_tau_outside', 'test_linear_L0', 'test_R1R2_0',
                'use_mlp', 'use_best_params', 'no_early',
                'add_source', 'symmetric_attention', 'sym_row_max','symmetric_QK',
                'diffusion', 'repulsion', 'drift', 'tau_residual',
                'XN_no_activation','m2_mlp', 'gnl_thresholding', 'gnl_W_param_free', 'gnl_W_param_free2', 'gnl_attention',
                'two_hops', 'time_dep_w', 'time_dep_struct_w',
                'greed_SL', 'greed_undir',
                'gcn_enc_dec', 'gcn_fixed', 'gcn_non_lin', 'gcn_symm', 'gcn_bias', 'gcn_mid_dropout']
    arg_intersect = list(set(opt.keys()) & set(tf_args))
    for arg in arg_intersect:
        str_tf = opt[arg]
        bool_tf = t_or_f(str_tf)
        opt[arg] = bool_tf
    return opt

def default_params():
    pass
    # args = parser.parse_args()
    # opt = vars(args)
    # return(opt)

