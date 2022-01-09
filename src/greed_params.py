
def greed_test_params(opt):
    opt['block'] = 'constant'
    opt['step_size'] = 0.1
    opt['time'] = 10
    opt['method'] = 'euler'
    opt['no_early'] = True
    opt['epoch'] = 5
    opt['function'] = 'greed'
    opt['self_loop_weight'] = 0
    opt['test_no_chanel_mix'] = True
    opt['test_omit_metric'] = True
    opt['test_mu_0'] = True
    opt['test_tau_remove_tanh'] = True
    opt['tau_reg'] = 5  # opt['attention_dim']
    if opt['test_tau_remove_tanh']:
        opt['test_tau_symmetric'] = True
        opt['tau_reg'] = 5
    else:
        opt['test_tau_symmetric'] = False
    return opt

def greed_run_params(opt):
    #fixed greed params
    # opt['function'] = 'greed'
    opt['block'] = 'constant'
    opt['method'] = 'euler'
    opt['no_early'] = True
    opt['epoch'] = 5 #10
    opt['self_loop_weight'] = 0
    return opt

def greed_hyper_params(opt):
    #tuning params
    opt['time'] = 5 #10
    opt['hidden_dim'] = 32 # 50 #60 #64 comment Karate needs a very high dimension to be able to learn
    opt['step_size'] = 0.1
    opt['tau_reg'] = 5
    return opt

def greed_ablation_params(opt):
    #ablation flags
    opt['test_no_chanel_mix'] = True
    opt['test_omit_metric'] = True
    opt['test_mu_0'] = True
    opt['test_tau_remove_tanh'] = False
    # opt['tau_reg'] = 5
    # opt['test_tau_remove_tanh_reg'] = 5  # opt['attention_dim']
    if opt['test_tau_remove_tanh']:
        opt['test_tau_symmetric'] = True
        # opt['test_tau_remove_tanh_reg'] = 5 found this has different tolerances
    else:
        opt['test_tau_symmetric'] = False
    return opt

def t_or_f(tf_str):
    if tf_str == "True" or tf_str == "true" or tf_str:
        return True
    else:
        return False

def tf_ablation_args(opt):
    for arg in ['test_no_chanel_mix','test_omit_metric','test_mu_0','test_tau_remove_tanh','test_tau_symmetric']:
        str_tf = opt[arg]
        bool_tf = t_or_f(str)
        opt[arg] = bool_tf
    return opt