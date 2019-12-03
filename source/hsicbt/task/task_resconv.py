from . import *


def plot_each_resconv_result(log_format, log_backprop, ext, fig_prefix, config_dict):

    # metadata = {
    #     #'title':'{} batch training performance'.format(config_dict['data_code']),
    #     'title': '',
    #     'xlabel': 'epochs',
    #     'ylabel': 'training batch accurarcy',
    #     'label': ['stadnard-train', 'format-train']
    # }

    # plot.plot_batches_log([log_backprop['batch_log_list'], log_format['batch_log_list']], 'batch_acc', metadata)
    # outpath = get_exp_path("{}-batch.{}".format(get_plot_filename(config_dict), ext))
    # save_experiment_fig(outpath)

    # metadata = {
    #     #'title':'{} training performance'.format(config_dict['data_code']),
    #     'title': '',
    #     'xlabel': 'epochs',
    #     'ylabel': 'training accurarcy',
    #     'label': ['backprop-train', 'format-train']
    # }
    # plot.plot_epoch_log([log_backprop['epoch_log_dict'], log_format['epoch_log_dict']], 'train_acc', metadata)
    # filepath = get_exp_path("{}-epoch-train-acc.{}".format(get_plot_filename(config_dict), ext))
    # save_experiment_fig(filepath)

    metadata = {
        #'title':'{} test performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epoch',
        'ylabel': 'test acc',
        'label': ['backprop', 'format']
    }
    plot.plot_epoch_log([log_backprop['epoch_log_dict'], log_format['epoch_log_dict']], 'test_acc', metadata)
    filepath = get_exp_path("{}-{}-epoch-test-acc.{}".format(fig_prefix, get_plot_filename(config_dict), ext))
    save_experiment_fig(filepath)

def plot_resconv_result(config_dict):

    
    try:
        log_format_mnist     = load_logs(get_log_filepath(config_dict['task'], TTYPE_FORMAT,    'mnist'  ))
        log_backprop_mnist   = load_logs(get_log_filepath(config_dict['task'], TTYPE_STANDARD , 'mnist'  ))
        log_format_cifar10   = load_logs(get_log_filepath(config_dict['task'], TTYPE_FORMAT,    'cifar10'))        
        log_backprop_cifar10 = load_logs(get_log_filepath(config_dict['task'], TTYPE_STANDARD , 'cifar10'))
        log_format_fmnist    = load_logs(get_log_filepath(config_dict['task'], TTYPE_FORMAT,    'fmnist' ))        
        log_backprop_fmnist  = load_logs(get_log_filepath(config_dict['task'], TTYPE_STANDARD , 'fmnist' ))
                        
    except IOError as e:
        print_highlight("{}.\nNo plot produced unless all backprop/format training has been done. (by altering \'training_type\' in config)".format(e), 'red')
        quit()
    config_dict['data_code'] = 'mnist' # sorry, i'm a bad programmer
    plot_each_resconv_result(log_format_mnist, log_backprop_mnist, config_dict['ext'], 'fig7a', config_dict)
    config_dict['data_code'] = 'cifar10'
    plot_each_resconv_result(log_format_cifar10, log_backprop_cifar10, config_dict['ext'], 'fig7b', config_dict)
    config_dict['data_code'] = 'fmnist'    
    plot_each_resconv_result(log_format_fmnist, log_backprop_fmnist, config_dict['ext'], 'fig7c', config_dict)        

def task_resconv_func(config_dict):
    func = task_assigner(config_dict['training_type'])
    func(config_dict)
        
