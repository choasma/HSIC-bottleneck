from . import *

def plot_variedact_result(config_dict):
    
    try:
        out_standard_batch_relu = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 1))['batch_log_list']
        out_standard_batch_tanh = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 2))['batch_log_list']
        out_standard_batch_elu = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 3))['batch_log_list']
        out_standard_batch_sigmoid = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 4))['batch_log_list']
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()


    input_list = [out_standard_batch_relu, out_standard_batch_tanh, out_standard_batch_elu, out_standard_batch_sigmoid]
    label_list = ['relu', 'tanh', 'elu', 'sigmoid']

    metadata = {
        #'title':'nHSIC(X, Z_L) of Varied-activation',
        'title': '',
        'xlabel': 'epoch',
        'ylabel': 'nHSIC($\mathbf{X}$, $\mathbf{Z}_L$)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
    filepath = get_exp_path("fig2a-varied-activation-hsic_xz-{}.{}".format(config_dict['data_code'], config_dict['ext']))
    save_experiment_fig(filepath)

    metadata = {
        #'title':'nHSIC(Y, Z_L) of Varied-activation',
        'title':'',
        'xlabel': 'epoch',
        'ylabel': 'nHSIC($\mathbf{Y}$, $\mathbf{Z}_L$)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
    filepath = get_exp_path("fig2b-varied-activation-hsic_yz-{}.{}".format(config_dict['data_code'], config_dict['ext']))
    save_experiment_fig(filepath)
    
    metadata = {
        #'title':'performance of Varied-activation',
        'title':'',
        'xlabel': 'epoch',
        'ylabel': 'train acc',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_acc', metadata)
    filepath = get_exp_path("fig2c-varied-activation-acc-{}.{}".format(config_dict['data_code'], config_dict['ext']))
    save_experiment_fig(filepath)


def task_variedact_func(config_dict):

    func = task_assigner(config_dict['training_type'])
    func(config_dict)


