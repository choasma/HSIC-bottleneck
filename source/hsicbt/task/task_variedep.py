from . import *

def plot_variedep_result(config_dict):
    try:
        out_standard_batch_001 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))['batch_log_list']
        out_standard_batch_005 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))['batch_log_list']
        out_standard_batch_010 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))['batch_log_list']
        out_standard_epoch_001 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))['epoch_log_dict']
        out_standard_epoch_005 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))['epoch_log_dict']
        out_standard_epoch_010 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))['epoch_log_dict']

    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    input_batch_list = [out_standard_batch_001, out_standard_batch_005, out_standard_batch_010]
    input_epoch_list = [out_standard_epoch_001, out_standard_epoch_005, out_standard_epoch_010]
    label_list = ['HTep=1', 'HTep=5', 'HTep=10']

    # metadata = {
    #     #'title':'HSIC(X, Z_L) of Varied-epoch',
    #     'title': '',
    #     'xlabel': 'epochs',
    #     'ylabel': 'HSIC(X, Z_L)',
    #     'label': label_list
    # }
    # plot.plot_batches_log(input_batch_list, 'batch_hsic_hx', metadata)
    # plot.save_figure(get_exp_path("varied-epoch-hsic_xz-{}.{}".format(
    #     config_dict['data_code'], config_dict['ext'])))

    # metadata = {
    #     #'title':'HSIC(Y, Z_L) of Varied-epoch',
    #     'title': '',
    #     'xlabel': 'epochs',
    #     'ylabel': 'HSIC(Y, Z_L)',
    #     'label': label_list
    # }
    # plot.plot_batches_log(input_batch_list, 'batch_hsic_hy', metadata)
    # plot.save_figure(get_exp_path("varied-epoch-hsic_yz-{}.{}".format(
    #     config_dict['data_code'], config_dict['ext'])))

    metadata = {
        #'title':'format-train of Varied-epoch',
        'title': '',
        'xlabel': 'epoch',
        'ylabel': 'train acc',
        'label': label_list
    }
    plot.plot_batches_log(input_batch_list, 'batch_acc', metadata)
    filepath = get_exp_path("fig5a-varied-epoch-acc-{}.{}".format(config_dict['data_code'], config_dict['ext']))
    save_experiment_fig(filepath)
    
    metadata = {
        #'title':'format-train of Varied-epoch',
        'title': '',
        'xlabel': '',
        'ylabel': '',
        'label': label_list
    }
    plot.plot_batches_log(input_batch_list, 'batch_loss', metadata)
    filepath = get_exp_path("fig5b-varied-epoch-loss-{}.{}".format(config_dict['data_code'], config_dict['ext']))
    save_experiment_fig(filepath)
    
    metadata = {
        #'title':'{} test performance of Varied-epoch'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epochs',
        'ylabel': 'test acc',
        'label': label_list
    }
    plot.plot_epoch_log(input_epoch_list, 'test_acc', metadata)
    plot.save_figure(get_exp_path("fig5a-{}-epoch-test-acc.{}".format(
        get_plot_filename(config_dict), config_dict['ext'])))
    
    metadata = {
        #'title':'{} test performance of Varied-epoc'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epochs',
        'ylabel': 'test loss',
        'label': label_list
    }
    plot.plot_epoch_log(input_epoch_list, 'test_loss', metadata)
    plot.save_figure(get_exp_path("fig5b-{}-epoch-test-loss.{}".format(
        get_plot_filename(config_dict), config_dict['ext'])))

    
def task_variedep_func(config_dict):

    model_filename = config_dict['model_file']
    func = task_assigner(config_dict['training_type'])
    config_dict['model_file'] = "{}-{:04d}.pt".format(
        os.path.splitext(model_filename)[0], config_dict['exp_index'])
    func(config_dict) 


