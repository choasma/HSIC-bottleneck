from . import *

def plot_hsicsolve_result(config_dict):

    try:
        out_epoch      = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code']))['epoch_log_dict']
        out_hsic_epoch = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code']))['epoch_log_dict']
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    metadata = {
        #'title':'{} test performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epoch',
        'ylabel': 'test acc',
        'label': ['backprop', 'unformat']
    }
    filename = ""
    plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'test_acc', metadata)
    filepath = get_exp_path("fig4-{}-test-acc.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    save_experiment_fig(filepath)

    metadata = {
        #'title':'{} test performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': '',
        'ylabel': '',
        'label': ['backprop', 'unformat']
    }
    
    # plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'train_acc', metadata)
    # filepath = get_exp_path("fig5-{}-train-acc.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    # save_experiment_fig(filepath)
    
    plot.plot_activation_distribution(get_act_path(config_dict['task'], config_dict['training_type'], config_dict['data_code']))
    filepath = get_exp_path("fig3-hsic-solve-actdist-{}.{}".format(config_dict['data_code'], config_dict['ext']))
    save_experiment_fig(filepath)

def task_hsicsolve_func(config_dict):

    if config_dict['do_training']:
        func = task_assigner(config_dict['training_type'])
        if config_dict['training_type'] == TTYPE_HSICTRAIN:
            config_dict['last_hidden_width'] = 10 # since we are using hsic to solve classification            
        func(config_dict)
