import argparse
import datetime

def greed_test_params(opt):
    opt['wandb_track_grad_flow'] = False
    opt['wandb_watch_grad'] = False
    opt['block'] = 'constant'
    opt['step_size'] = 0.1
    opt['time'] = 10
    opt['method'] = 'euler'
    opt['no_early'] = True
    opt['epoch'] = 5
    opt['function'] = 'greed'
    opt['self_loop_weight'] = 0

    #set all flags to False to hit the tests
    opt['test_no_chanel_mix'] = False
    opt['test_omit_metric'] = False
    opt['test_mu_0'] = False
    opt['test_tau_remove_tanh'] = False
    opt['tau_reg'] = 5  # opt['attention_dim']
    if opt['test_tau_remove_tanh']:
        opt['test_tau_symmetric'] = True
        opt['tau_reg'] = 5
    else:
        opt['test_tau_symmetric'] = False
    opt['test_tau_outside'] = False

    opt['test_R1R2_0'] = False
    opt['test_grand_metric'] = False
    opt['test_tau_ones'] = False
    opt['symmetric_QK'] = False
    opt['symmetric_attention'] = False
    opt['sym_row_max'] = False

    opt['greed_momentum'] = False
    opt['gamma_epsilon'] = 1e-3
    return opt

def greed_run_params(opt):
    #run params
    opt['block'] = 'constant'
    opt['self_loop_weight'] = 0.0 #1.0 #0 #needed for greed
    opt['no_early'] = True #False #- this works as an override of best param as only pubmed has this key
    return opt

def greed_hyper_params(opt):
    #ablation flags
    opt['test_no_chanel_mix'] = True #True
    opt['test_tau_remove_tanh'] = False #True
    opt['test_tau_symmetric'] = True#False

    #if function is greed_linear
    opt['test_linear_L0'] = True # flag to make the Laplacian form only dependent on embedding not time #redundant flag should be assumed in greed_linear_{homo, hetro}
    opt['test_R1R2_0'] = True
    # opt['test_grand_metric'] = True #old parameter

    #greed_linear_homo params
    opt['symmetric_attention'] = True
    opt['attention_type'] = "scaled_dot" #'scaled_dot'
    opt['attention_normalisation'] = 'none'

    #greed_linear_hetero params
    opt['symmetric_QK'] = False #True #False
    opt['attention_activation'] = 'exponential'#'softmax' #, exponential

    opt['T0term_normalisation'] = 'T0_rowSum'
    opt['laplacian_norm'] = 'lap_symmDeg_RowSumnorm' #'lap_symmAtt_relaxed' #'lap_symmDeg_RowSumnorm'#'lap_symmDeg_RowSumnorm' #'lap_symmAtt_RowSumnorm' #, lap_symmAttM_RowSumnorm
    # opt['R_T0term_normalisation'] = 'T0_rowSum'
    # opt['R_laplacian_norm'] = 'lap_symmAtt_relaxed' #'lap_symmAtt_relaxed' #lap_symmAtt_RowSumnorm' #, lap_symmAttM_RowSumnorm
    opt['R_depon_A'] = 'none' #'inverse'
    opt['alpha_style'] = 'sigmoid' #0.5 'sigmoid' #"sigmoid", "free", "forced", "diag"

    #hetero experiment flags
    opt['test_omit_metric_L'] = False #True
    opt['test_omit_metric_R'] = False #True
    opt['test_tau_ones'] = True #False #True  ####<- this is key for hetero datasets
    opt['test_tau_symmetric'] = False #False #False
    opt['W_type'] = 'identity'
    opt['R_W_type'] = 'identity'
    opt['tau_residual'] = True

    opt['test_mu_0'] = True # False #True
    opt['add_source'] = True #False #True
    opt['XN_no_activation'] = True #False
    opt['use_mlp'] = False #True #encoder mlp
    opt['m2_mlp'] = False #False #decoder mlp
    opt['beltrami'] = False
    # opt['pos_enc_type'] = 'GDC'

    #hyper-params
    opt['optimizer'] = 'adam'
    opt['lr'] = 0.001#0.001
    opt['dropout'] = 0.35
    opt['input_dropout'] = 0.5
    opt['decay'] = 0.0005#005 #Cora 0.05 chameleon 0.0005
    opt['hidden_dim'] = 64#64 #512
    opt['use_best_params'] = False #True #False #True
    opt['method'] = 'euler'
    opt['max_nfe'] = 5000 #for some reason 1000 not enough with all report building
    opt['step_size'] = 0.5 #1.0 #0.1 #have changed this to 0.1  dafault in run_GNN.py
    opt['time'] = 3 #4 #18.295 #10
    opt['epoch'] = 129#257#129 #20#6#129 #6#9#129 #255#129 #254 #100 #40 #40 #10
    opt['num_splits'] = 1 #10#4#1
    opt['use_labels'] = False #True
    # opt['planetoid_split'] = True
    # opt['geom_gcn_splits'] = False #True#True #False#True
    # opt['patience'] = 3
    # opt['target_homoph'] = '0.70' #for synthetic cora

    #greed_non_linear params
    opt['two_hops'] = False # This turns on the functionality to get equation 28 working
    opt['time_dep_unstruct_w'] = False#False#True
    opt['time_dep_struct_w'] = True#False#True
    assert not(opt['time_dep_unstruct_w'] and opt['time_dep_struct_w']), "can't do both"
    opt['gnl_style'] = 'general_graph'#'att_rep_laps'#'att_rep_laps' #'general_graph'#'softmax_attention' #'general_graph'#'scaled_dot' #'softmax_attention' #'scaled_dot'
    opt['gnl_measure'] = 'ones'#'nodewise' #'deg_poly' #'ones' #'deg_poly' # 'nodewise'

    if opt['gnl_style'] == 'scaled_dot':
        opt['gnl_omega'] = 'diag' #'attr_rep' #'sum' #'attr_rep' #'attr_rep' #'attr_rep' #'sum'  # 'product' # 'product'  #method to make Omega symmetric
        opt['dim_p_w'] = 16 #4
        opt['gnl_activation'] = 'squareplus_deriv' # exponential sigmoid_deriv tanh_deriv, squareplus_deriv
        opt['gnl_omega_norm'] = 'tanh' #"rowSum"
    elif opt['gnl_style'] == 'softmax_attention':
        opt['symmetric_attention'] = True #should be redundant
        opt['attention_type'] = "scaled_dot"
        opt['symmetric_QK'] = True
        opt['attention_activation'] = 'softmax'#'softmax' #, exponential
        opt['attention_normalisation'] = 'none'
    elif opt['gnl_style'] == 'general_graph':
        opt['gnl_activation'] = 'identity'#'sigmoid' #'identity'
        opt['gnl_attention'] = False #use L0 attention coefficients
        #Omega
        opt['gnl_omega'] = 'diag'#'Omega_W_eig'#'diag' #'diag'#'zero' Omega_eq_W
        opt['gnl_omega_diag'] = 'free' #'free 'const'
        opt['gnl_omega_diag_val'] = None #1 #-1 # 1
        opt['gnl_omega_activation'] = 'identity' #identity
        # opt['gnl_omega_params'] = ["diag","free","None","identity"] #[opt['gnl_omega'], opt['gnl_omega_diag'], opt['gnl_omega_diag_val'], opt['gnl_omega_activation']]
        #W
        #explanations of z2x W methods:
        # 'Z_diag' - initialise as 'sum' but then set W as eval and use z2x
        # 'GS_Z_diag' - do gram schmidt for evec set W as eval and use z2x
        # 'cgnn_Z_diag' - do cgnn update on evec
        # 'loss_W_orthog', - free sum tensor set W as eval and use z2x
        # 'W_orthog_init', - use GS to init param W_U as orthog - then set W=ULU.T - set W as eval and use z2x
        # 'householder' - use householder reflections to enforce orthog W_U
        # 'skew_sym' - use skew-symetric and bilinear approximation to enforce orthog W_U
        opt['gnl_W_style'] = 'sum'#'householder'#'skew_sym'#'cgnn_Z_diag'#'W_orthog_init'#'cgnn_Z_diag'#'loss_W_orthog'#'cgnn_Z_diag'#'diag_dom'#'GS_Z_diag'#'diag_dom'#'Z_diag'#'sum'#'diag_dom'#'diag_dom'#'sum'#'neg_prod'#'sum'#'diag_dom' #'sum' #'diag_dom'#'k_diag_pc'#'diag_dom'  # 'sum' #'k_diag'#'k_block' #'diag_dom' # 'cgnn'#'GS'#sum, prod, GS, cgnn
        if opt['gnl_W_style'] == 'k_block':
            assert opt['hidden_dim'] % opt['k_blocks'] == 1 and opt['k_blocks'] * opt['block_size'] <= opt['hidden_dim']#in_features, 'must have odd number of k diags'
            opt['k_blocks'] = 2#1
            opt['block_size'] = 5
        elif opt['gnl_W_style'] == 'k_diag':
            assert opt['k_diags'] % 2 == 1 and opt['k_diags'] <= opt['hidden_dim']#, 'must have odd number of k diags'
            opt['k_diags'] = 13
        elif opt['gnl_W_style'] in ['diag', 'diag_dom']:
            opt['gnl_W_diag_init'] = 'uniform'#'identity'
            opt['gnl_W_param_free'] = 'True' #'True'
            opt['gnl_W_diag_init_q'] = 1.0
            opt['gnl_W_diag_init_r'] = 0.0
        elif opt['gnl_W_style'] == 'householder':
            opt['householder_L'] = opt['hidden_dim'] - 1 #63

    elif opt['gnl_style'] == 'att_rep_laps':
        opt['gnl_W_style'] = 'att_rep_lap_block'#'sum'#'att_rep_lap_block'
    opt['drift'] = False  # False#True
    opt['gnl_thresholding'] = False
    opt['drift_space'] = 'label' #feature' #'label'
    opt['drift_grad'] = False#True #True
    opt['m2_aug'] = False #True #False #reads out (no weights) prediction from bottom C dimensions
    opt['m1_W_eig'] = False#True
    opt['m2_W_eig'] = 'z2x' #False #'x2z'#'z2x' #True #True
    # if opt['m2_W_eig'] == 'z2x': #not true as can just do eigen decomp for sum for example
    #     assert opt['gnl_W_style'] in ['Z_diag', 'GS_Z_diag', 'cgnn_Z_diag', 'loss_W_orthog', 'W_orthog_init', 'householder', 'skew_sym'], 'z2x must have diag style matrix'
    opt['m3_path_dep'] = None#'label_att'#'feature_jk'#None#'label_jk'#'train_centers'#'feature_jk'#'label_jk'#'feature_jk' #'label_jk' 'label_att'
    opt['path_dep_norm'] = None#'rayleigh1' #'z_cat_normed_z'#'rayleigh'#'nodewise' #'rayleigh'
    # opt['m3_best_path_dep'] = False #todo add to params - makes prediction using path of train set evolution/performance
    # opt['m3_space'] = None
    opt['loss_reg'] = None #4#3#2#6#5#6
    opt['loss_reg_weight'] = 1. #4
    opt['loss_reg_delay'] = 0 #4
    opt['loss_reg_certainty'] = None #0.85 #0.95 #1.00 #0.95
    opt['dampen_gamma'] = 1.0#0.6    #assuming spec rad=4, dampen gamma=0.6, step=0.1
    opt['gnl_W_norm'] = False#True#False  # True #divide by spectral radius
    opt['pointwise_nonlin'] = False#True#False#True #todo add to args
    opt['loss_orthog_a'] = 0#1.0

    # att_rep_laplacians
    opt['diffusion'] = True#True
    opt['repulsion'] = True#False

    #definitions of lie trotter
    #None - runs greed_non_linear with diffusion with optional simultaneous drift (ie eq 40) and the potential to pseudo inverse threshold
    #gen_0 - alternates one step diffusion and drift in alternating lie-trotter scheme (ie eq 42)
    #gen_1 - alternates ranges of diffusion and drift (ie eq 43-44)
    #gen_2 - rolls out blocks of diffusion/drift/thresholding/label diffusion - using function_greed_non_linear_lie_trotter.py
    # reports_list = ['spectrum', 'acc_entropy', 'edge_evol', 'node_evol', 'node_scatter', 'edge_scatter', 'class_dist' ,'TSNE', 'val_test_entropy']
    opt['reports_list'] = []#[1,2,4,7,8,9,10]#,11] #[1,2,4,7,8,9,10,11]#[1,2,3,4,5,6,7,8,9]#[1,2,4,7,8,9]#] #[8]#[1,2,4,5,7,8]  # [1]#[1,2,3,4,5,6,7] #
    opt['lie_trotter'] = None#'gen_2' #'gen_2' #'gen_2' #'gen_2' #None #'gen_2'#'gen_1' #'gen_0' 'gen_1' 'gen_2'
    if opt['lie_trotter'] in [None, 'gen_0', 'gen_1']:
        if opt['lie_trotter'] in [None, 'gen_0']:
            opt['threshold_times'] = [2,4] #takes an euler step that would have been taken in drift diffusion and also thresholds between t->t+1
        elif opt['lie_trotter'] == 'gen_1':
            #gen1 args
            opt['diffusion_ranges'] = [[0,2],[3,5]]
            opt['drift_ranges'] = [[2,3],[5,6]]
            #solver args
            opt['time'] = 6.0
            opt['step_size'] = 1.0
            opt['method'] = 'euler'
    elif opt['lie_trotter'] == 'gen_2': #todo test if time dep W works with gen 2 - and then implement the W block end transform (maybe rewiring also) idea
        opt['block'] = 'greed_lie_trotter'
        #gen2 args
        #todo we only ever want one decoder, do this by setting 'share_block'=0 (to the number of the first block for any drift blocks)
        # opt['lt_gen2_args'] = [{'lt_block_type': 'diffusion', 'lt_block_time': 2, 'lt_block_step': 1.0, 'lt_block_dimension': 256, 'share_block': None, 'reports_list': [1]},
        #                     {'lt_block_type': 'drift', 'lt_block_time': 1, 'lt_block_step': 1.0, 'lt_block_dimension': 256, 'share_block': 0, 'reports_list': []},
        #                     {'lt_block_type': 'diffusion', 'lt_block_time': 2, 'lt_block_step': 1.0,'lt_block_dimension': 256, 'share_block': None, 'reports_list': [1]},
        #                     {'lt_block_type': 'drift', 'lt_block_time': 1, 'lt_block_step': 1.0, 'lt_block_dimension': 256, 'share_block': 0, 'reports_list': []},
        #                     {'lt_block_type': 'label', 'lt_block_time': 2, 'lt_block_step': 1.0, 'lt_block_dimension': 256, 'share_block': None, 'reports_list': [1,2,3,4,5,6,7]}]
        #double diffusion config
        # opt['lt_gen2_args'] = [{'lt_block_type': 'diffusion', 'lt_block_time': 3, 'lt_block_step': 0.5, 'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': []},
        #                        {'lt_block_type': 'diffusion', 'lt_block_time': 3, 'lt_block_step': 0.5, 'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': []},
        #                        {'lt_block_type': 'diffusion', 'lt_block_time': 3, 'lt_block_step': 0.5, 'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': [1,2,4,7,8,9,10]}]#,
                               # {'lt_block_type': 'drift', 'lt_block_time': 1, 'lt_block_step': 0.5, 'lt_block_dimension': opt['hidden_dim'], 'share_block': 0, 'reports_list': [1,2,4,7,8,9,10]}]#,
                               # {'lt_block_type': 'diffusion', 'lt_block_time': 3, 'lt_block_step': 1.0, 'lt_block_dimension': opt['hidden_dim'], 'share_block': None, 'reports_list': []}]#[1,2,3,4,5,6,7]}]#[]}]
        #for "restart" diffusion
        opt['time2'] = 2.0
        opt['time3'] = 1.0

    #gcn params
    # opt['function'] = 'gcn_dgl'#'gcn_res_dgl' #'gcn_dgl'#'greed_non_linear' #'gcn' #'greed_non_linear' #'greed_linear_hetero'
    # opt['gcn_enc_dec'] = False #False #True
    # opt['gcn_fixed'] = False #False #True
    # opt['gcn_symm'] = False#True
    # opt['gcn_non_lin'] = True #False #True
    # opt['gcn_bias'] = True
    # opt['gcn_mid_dropout'] = False
    # opt['gcn_params'] = [5, 'gcn_res_dgl', True, True, True, False]#[0, 'gcn_dgl', False, False, False, True]#[1, 'gcn_dgl', True, False, False, True]#[2, 'gcn_res_dgl', True, False, False, True]#[3, 'gcn_res_dgl', True, True, False, True]#[4, 'gcn_res_dgl', True, True, True, True]#[5, 'gcn_res_dgl', True, True, True, False]

    return opt

def not_sweep_args(opt, project_name, group_name):
    opt['gnl_savefolder'] = "animation_runs"#'tsne_evol'#'chameleon_general_drift'#'chameleon_testing'
    opt['wandb_entity'] = "graph_neural_diffusion"
    opt['wandb_project'] = "animation_runs"#"reporting_runs_drift"
    opt['wandb_group'] = "reporting_group" #"ablation_group"#"reporting_group"
    # opt['wandb_project'] = project_name #"greed_runs"
    # opt['wandb_group'] = group_name #"testing"  # "tuning" eval

    # args for running locally - specified in YAML for tunes
    opt['wandb'] = True #True #False #True
    opt['wandb_track_grad_flow'] = False#True #False  #collect stats for reports
    opt['run_track_reports'] = False#True #False#True ##run the evolution reports
    opt['save_local_reports'] = True#True
    opt['save_wandb_reports'] = True#False#True
    opt['wandb_watch_grad'] = False

    opt['wandb_epoch_list'] = [1,4,8,16,32,64,96,128]#[8, 128]#[1,2,3,4] #[1,2,4,8,16,32,64,128]#[1,2,3,4,5]#,6,7,8]
    opt['display_epoch_list'] = [16,128]#[1,2,3,4] #[1,2,4,8,16,32,64,128]#[1,2,3,4,5]#,6,7,8] #todo add to params

    DT = datetime.datetime.now()
    opt['wandb_run_name'] = DT.strftime("%m%d_%H%M%S_") + "wandb"#
    # hyper-params
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
                'use_mlp', 'use_best_params', 'no_early', 'use_labels', 'hetero_SL', 'hetero_undir',
                'add_source', 'symmetric_attention', 'sym_row_max','symmetric_QK',
                'diffusion', 'repulsion', 'drift', 'tau_residual',
                'XN_no_activation','m2_mlp', 'gnl_thresholding', 'gnl_W_param_free', 'gnl_W_param_free2', 'gnl_attention',
                'two_hops', 'time_dep_unstruct_w', 'time_dep_struct_w',
                'greed_SL', 'greed_undir', 'm2_aug', 'm1_W_eig', 'gnl_W_norm', 'drift_grad', 'pointwise_nonlin',
                'gcn_enc_dec', 'gcn_fixed', 'gcn_non_lin', 'gcn_symm', 'gcn_bias', 'gcn_mid_dropout',
                'wandb', 'wandb_sweep', 'adjoint']
    arg_intersect = list(set(opt.keys()) & set(tf_args))
    for arg in arg_intersect:
        str_tf = opt[arg]
        bool_tf = t_or_f(str_tf)
        opt[arg] = bool_tf

    return opt

def default_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
    # parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--use_labels', type=str, default='False', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from '
                             'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                        help='the number of splits to repeat the results on')
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
    parser.add_argument('--patience', type=int, default=None, help='set if training should use patience on val acc')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, help='laplacian, transformer, greed, GAT')
    # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
    #                     help='Add a fully connected layer to the encoder.')
    # parser.add_argument('--add_source', dest='add_source', action='store_true',
    #                     help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=0.1,
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
    # parser.add_argument("--no_early", action="store_true",
    #                     help="Whether or not to use early stopping of the ODE integrator when testing.")
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
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

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
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                        help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                        help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                        help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    # parser.add_argument('--symm_QK', action='store_true',
    #                     help='makes the attention symmetric for rewring in QK space')
    # parser.add_argument('--symmetric_attention', action='store_true',
    #                     help='makes the attention symmetric via (A+A.T)/2')#for rewring in QK space')
    # parser.add_argument('--sym_row_max', action='store_true',
    #                     help='makes every row sum less than 1 by dividing by max rum some')
    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")
    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

    # wandb logging and tuning
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('-wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="graph_neural_diffusion", type=str,
                        help="jrowbottomwnb, ger__man")  # not used as default set in web browser settings
    parser.add_argument('--wandb_project', default="greed", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 254],
                        help='list of epochs to log gradient flow, 1 based')
    parser.add_argument('--run_track_reports', action='store_true', help="run_track_reports")
    parser.add_argument('--save_wandb_reports', action='store_true', help="save_wandb_reports")
    parser.add_argument('--save_local_reports', action='store_true', help="save_local_reports")

    # wandb setup sweep args
    parser.add_argument('--tau_reg', type=int, default=2)
    parser.add_argument('--test_mu_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_no_chanel_mix', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric_L', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric_R', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_remove_tanh', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_symmetric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_outside', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_linear_L0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_R1R2_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_grand_metric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_ones', type=str, default='True')  # action='store_true')

    # Temp changing these to be strings so can tune over
    parser.add_argument('--use_mlp', type=str, default='False')  # action='store_true')
    parser.add_argument('--add_source', type=str, default='False',
                        help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--no_early', type=str, default='False')  # action='store_true')
    parser.add_argument('--symmetric_QK', type=str, default='False',
                        help='makes the attention symmetric for rewring in QK space')
    parser.add_argument('--symmetric_attention', type=str, default='False',
                        help='makes the attention symmetric via (A+A.T)/2')  # for rewring in QK space')
    parser.add_argument('--sym_row_max', type=str, default='False',
                        help='makes every row sum less than 1 by dividing by max rum some')

    # greed args
    parser.add_argument('--use_best_params', action='store_true', help="flag to take the best BLEND params")
    parser.add_argument('--greed_momentum', action='store_true', help="flag to use momentum grad flow")
    parser.add_argument('--momentum_alpha', type=float, default=0.2, help="alpha to use in momentum grad flow")
    parser.add_argument('--dim_p_omega', type=int, default=16, help="inner dimension for Omega")
    parser.add_argument('--dim_p_w', type=int, default=16, help="inner dimension for W")
    parser.add_argument('--gamma_epsilon', type=float, default=0.01,
                        help="epsilon value used for numerical stability in get_gamma")

    #needed for run_best_sweeps
    parser.add_argument('--sweep_id', type=str, default='', help="sweep_id for 1 best run")  # action='store_true')
    parser.add_argument('--run_id', type=str, default='', help="run_id for 1 best run")  # action='store_true')
    parser.add_argument('--run_group', type=str, default=None, help="run_id for 1 best run")  # action='store_true')

    parser.add_argument('--XN_no_activation', type=str, default='False',
                        help='whether to relu activate the terminal state')
    parser.add_argument('--m2_mlp', type=str, default='False', help='whether to use decoder mlp')
    parser.add_argument('--attention_activation', type=str, default='exponential',
                        help='[exponential, sigmoid] activations for the GRAM matrix')
    parser.add_argument('--attention_normalisation', type=str, default='sym_row_col',
                        help='[mat_row_max, sym_row_col, row_bottom, "best"] how to normalise')
    parser.add_argument('--T0term_normalisation', type=str, default='T0_identity',
                        help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
    parser.add_argument('--laplacian_norm', type=str, default='lap_noNorm',
                        help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')
    parser.add_argument('--R_T0term_normalisation', type=str, default='T0_identity',
                        help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
    parser.add_argument('--R_laplacian_norm', type=str, default='lap_noNorm',
                        help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')

    parser.add_argument('--alpha_style', type=str, default='', help='"sigmoid", "free", "forced", "matrix"')
    parser.add_argument('--fix_alpha', type=float, default=None, help='control balance between diffusion and repulsion')
    parser.add_argument('--diffusion', type=str, default='True', help='turns on diffusion')
    parser.add_argument('--repulsion', type=str, default='False', help='turns on repulsion')
    # parser.add_argument('--drift', type=str, default='False', help='turns on drift')
    parser.add_argument('--W_type', type=str, default='identity', help='identity, diag, full')
    parser.add_argument('--R_W_type', type=str, default='identity', help='for repulsion: identity, diag, full')
    parser.add_argument('--R_depon_A', type=str, default='', help='R dependancy in A')
    parser.add_argument('--W_beta', type=float, default=0.5, help='for cgnn Ws orthoganal update')
    parser.add_argument('--tau_residual', type=str, default='False', help='makes tau residual')

    parser.add_argument('--drift', type=str, default='False', help='turns on drift')
    parser.add_argument('--gnl_thresholding', type=str, default='False', help='turns on pseudo inverse thresholding')
    parser.add_argument('--lie_trotter', type=str, default=None, help='None, gen_0, gen_1, gen_2')
    parser.add_argument('--time2', type=float, default=None, help='LT2 block2 - End time of ODE integrator.')
    parser.add_argument('--time3', type=float, default=None, help='LT2 block3 - End time of ODE integrator.')

    # GCN ablation args
    parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
    parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
    parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
    parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
    parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')
    parser.add_argument('--gcn_mid_dropout', type=str, default='False', help='dropout between GCN layers')
    parser.add_argument('--gcn_params', nargs='+', default=None, help='list of args for gcn ablation')
    parser.add_argument('--gcn_params_idx', type=int, default=0, help='index to track GCN ablation')

    # greed non linear args
    parser.add_argument('--gnl_style', type=str, default='scaled_dot', help='scaled_dot, softmax_attention, general_graph')
    parser.add_argument('--gnl_activation', type=str, default='identity', help='identity, sigmoid, ...')
    parser.add_argument('--gnl_measure', type=str, default='ones', help='ones, deg_poly, nodewise')
    parser.add_argument('--gnl_omega', type=str, default='zero', help='zero, diag, sum')
    parser.add_argument('--gnl_omega_diag', type=str, default='free', help='free, const')
    parser.add_argument('--gnl_omega_activation', type=str, default='identity', help='identity, exponential')
    parser.add_argument('--gnl_W_style', type=str, default='sum', help='sum, prod, GS, cgnn, diag_dom')
    parser.add_argument('--gnl_attention', type=str, default='False', help='turns on attention for gnl general graph')

    parser.add_argument('--k_blocks', type=int, default=5, help='k_blocks')
    parser.add_argument('--block_size', type=int, default=5, help='block_size')
    parser.add_argument('--k_diags', type=float, default=11, help='k_diags')
    parser.add_argument('--k_diag_pc', type=float, default=0.1, help='percentage or dims diagonal')
    parser.add_argument('--gnl_W_diag_init', type=str, default='identity', help='init of diag elements [identity, uniform, linear]')
    parser.add_argument('--gnl_W_param_free', type=str, default='True', help='allow parameter to require gradient')
    parser.add_argument('--gnl_W_diag_init_q', type=float, default=1.0, help='slope of init of spectrum of W')
    parser.add_argument('--gnl_W_diag_init_r', type=float, default=0.0, help='intercept of init of spectrum of W')
    parser.add_argument('--gnl_W_norm', type=str, default='False', help='divide W matrix by its spectral radius')
    parser.add_argument('--two_hops', type=str, default='False', help='flag for 2-hop energy')
    parser.add_argument('--time_dep_unstruct_w', type=str, default='False', help='Learn a time dependent potentials')
    parser.add_argument('--time_dep_struct_w', type=str, default='False', help='Learn a structured time dependent potentials')
    parser.add_argument('--target_homoph', type=str, default='0.80', help='target_homoph for syn_cora [0.00,0.10,..,1.00]')
    parser.add_argument('--hetero_SL', type=str, default='True', help='control self loops for Chameleon/Squirrel')
    parser.add_argument('--hetero_undir', type=str, default='True', help='control undirected for Chameleon/Squirrel')
    parser.add_argument('--gnl_savefolder', type=str, default='', help='ie ./plots/{chamleon_gnlgraph_nodrift}')

    parser.add_argument('--omega_params', nargs='+', default=None, help='list of Omega args for ablation')
    parser.add_argument('--W_params', nargs='+', default=None, help='list of W args for ablation')
    parser.add_argument('--greed_params', nargs='+', default=None, help='list of args for focus models')

    parser.add_argument('--loss_reg', type=int, default=None, help='1-6')
    parser.add_argument('--loss_reg_weight', type=float, default=1.0, help='weighting for loss reg term')
    parser.add_argument('--loss_reg_delay', type=int, default=0.0, help='num epochs epochs to wait before applying loss reg')
    parser.add_argument('--loss_reg_certainty', type=float, default=1.0, help='amount of certainty to encode in prediction')

    parser.add_argument('--m2_aug', type=str, default='False', help='whether to augment m2 for drift readout')
    parser.add_argument('--m1_W_eig', type=str, default='False', help='project encoding onto W eigen basis')
    parser.add_argument('--m2_W_eig', type=str, default='', help='either z2x or x2z, project onto W eigen basis before decode')
    parser.add_argument('--m3_path_dep', type=str, default='', help='whether to use path dependent for m3 decoder')
    parser.add_argument('--path_dep_norm', type=str, default='', help='whether to norm the path dependent solution for m3 decoder')
    parser.add_argument('--drift_space', type=str, default=None, help='feature, label')
    parser.add_argument('--drift_grad', type=str, default='True', help='collect gradient off drift term')
    parser.add_argument('--dampen_gamma', type=float, default=1.0, help='gamma dampening coefficient, 1 is turned off, 0 is full dampening')
    parser.add_argument('--pointwise_nonlin', type=str, default='False', help='pointwise_nonlin')
    parser.add_argument('--loss_orthog_a', type=float, default=0, help='loss orthog term')
    parser.add_argument('--householder_L', type=int, default=8, help='num iterations of householder reflection for W_orthog')


    args = parser.parse_args()
    opt = vars(args)



    args = parser.parse_args()
    opt = vars(args)
    return(opt)

