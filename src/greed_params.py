
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
    opt['greed_momentum'] = False

    return opt

def greed_run_params(opt):
    #fixed greed params - handled by merge_cmd_args
    opt['function'] = 'greed'
    opt['block'] = 'constant'
    opt['self_loop_weight'] = 0 #needed for greed
    # opt['method'] = 'euler' #'dopri5' #'dopri5' #'euler' #need to comment this out for tuning
    # TUNING
    # opt['step_size'] = 1.0 #0.1 #have changed this to 0.1  dafault in run_GNN.py
    # opt['time'] = 10 #18.295 #10
    opt['epoch'] = 30 #10
    opt['no_early'] = True #False #- this works as an override of best param as only pubmed has this key

    #at some point test these  - not  so won't overwrite
    # not in best aprams
    # opt['greed_momentum'] = False #new param not in best_params
    # handled by merge_cmd_args
    # opt['add_source'] = False #this feels important because of design, ie keep adding back initial condition of the energy grad flow!
    return opt

def greed_hyper_params(opt):
    #tuning params
    opt['hidden_dim'] = 80 # 50 #60 #64 comment Karate needs a very high dimension to be able to learn
    opt['attention_dim'] = 32
    opt['tau_reg'] = 8

    opt["optimizer"] = 'adamax' #tune.choice(["adam", "adamax"]) #parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    opt["decay"] = 0.005 #tune.loguniform(0.001, 0.1)  # parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    opt["lr"] = 0.02292 #tune.uniform(0.01, 0.2) parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    opt["input_dropout"] = 0.5 #parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    opt["dropout"] = 0.04687 #tune.uniform(0, 0.15)  # output dropout parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')

    # opt["use_mlp"] = True### this was ignored in Sweeep as string from YAML

    # alpha = 1
    # heads = 8
    # method = 'dopri5'
    # use_mlp = False
    return opt

def greed_ablation_params(opt):
    #ablation flags
    opt['test_no_chanel_mix'] = True
    opt['test_omit_metric'] = True
    opt['test_mu_0'] = True
    opt['test_tau_remove_tanh'] = True
    # opt['tau_reg'] = 5
    # opt['test_tau_remove_tanh_reg'] = 5  # opt['attention_dim']
    if opt['test_tau_remove_tanh']:
        opt['test_tau_symmetric'] = True
        # opt['test_tau_remove_tanh_reg'] = 5 found this has different tolerances
    else:
        opt['test_tau_symmetric'] = True
    return opt

def t_or_f(tf_str):
    if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
        return True
    else:
        return False

def tf_ablation_args(opt):
    for arg in ['test_no_chanel_mix','test_omit_metric','test_mu_0','test_tau_remove_tanh','test_tau_symmetric',
                'use_mlp', 'add_source']:
        str_tf = opt[arg]
        bool_tf = t_or_f(str)
        opt[arg] = bool_tf
    return opt