
def greed_test_params(opt):
    opt['block'] = 'constant'
    opt['step_size'] = 0.1
    opt['time'] = 10
    opt['method'] = 'euler'
    opt['no_early'] = True

    opt['epoch'] = 5
    opt['function'] = 'greed'
    # opt['dataset'] = 'Karate'
    opt['dataset'] = 'Cora'

    opt['self_loop_weight'] = 0

    opt['pos_enc_hidden_dim'] = 16
    opt['feat_hidden_dim'] = 64  # 50 #60 #64 comment Karate needs a very high dimension to be able to learn
    opt['hidden_dim'] = opt['pos_enc_hidden_dim'] + opt['feat_hidden_dim']  # 64 #80 #8

    # added to test_params.py
    opt['test_no_chanel_mix'] = True
    opt['test_omit_metric'] = True
    opt['test_mu=0'] = True
    opt['test_tau_remove_tanh'] = False
    opt['test_tau_remove_tanh_reg'] = 5  # opt['attention_dim']

    if opt['test_tau_remove_tanh']:
        opt['test_tau_symmetric'] = True
        opt['test_tau_remove_tanh_reg'] = 5
    else:
        opt['test_tau_symmetric'] = False

    return opt