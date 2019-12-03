from . import *


def plot_general_result(config_dict):

    
    try:

        log_format = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code']))        
        log_backprop = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code']))

    except IOError as e:
        print_highlight("{}.\nNo plot produced unless all backprop/format training has been done. (by altering \'training_type\' in config)".format(e), 'red')
        quit()

    metadata = {
        #'title':'{} batch training performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': '',
        'ylabel': '',
        'label': ['backprop-train', 'format-train']
    }

    plot.plot_batches_log([log_backprop['batch_log_list'], log_format['batch_log_list']], 'batch_acc', metadata)
    outpath = get_exp_path("{}-batch.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    save_experiment_fig(outpath)

    metadata = {
        #'title':'{} training performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': '',
        'ylabel': '',
        'label': ['backprop-train', 'format-train']
    }
    plot.plot_epoch_log([log_backprop['epoch_log_dict'], log_format['epoch_log_dict']], 'train_acc', metadata)
    filepath = get_exp_path("{}-epoch-train-acc.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    save_experiment_fig(filepath)

    metadata = {
        #'title':'{} test performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': '',
        'ylabel': '',
        'label': ['backprop-train', 'format-train']
    }
    plot.plot_epoch_log([log_backprop['epoch_log_dict'], log_format['epoch_log_dict']], 'test_acc', metadata)
    filepath = get_exp_path("{}-epoch-test-acc.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    save_experiment_fig(filepath)

def task_general_func(config_dict):
    func = task_assigner(config_dict['training_type'])
    func(config_dict)
        
