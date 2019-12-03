from . import *

def plot_sigmacomb_result(config_dict):

    try:
        out_standard_batch_1 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))['batch_log_list']
        out_standard_batch_2 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))['batch_log_list']
        out_standard_batch_3 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))['batch_log_list']
        out_standard_batch_4 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 4))['batch_log_list']
        out_standard_epoch_1 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))['epoch_log_dict']
        out_standard_epoch_2 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))['epoch_log_dict']
        out_standard_epoch_3 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))['epoch_log_dict']
        out_standard_epoch_4 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 4))['epoch_log_dict']
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    input_list = [out_standard_batch_1, out_standard_batch_2, out_standard_batch_3, out_standard_batch_4]
    label_list = ['$\sigma$=1', '$\sigma$=5', '$\sigma$=10', '$\sigma$-combined']
    metadata = {
        #'title':'{} training perf of sigma-combined net'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epoch',
        'ylabel': 'train acc',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_acc', metadata)
    filepath = get_exp_path("fig6b-{}-sigmacomb-train-acc.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    save_experiment_fig(filepath)
        
    input_list = [out_standard_epoch_1, out_standard_epoch_2, out_standard_epoch_3, out_standard_epoch_4]
    label_list = ['$\sigma$=1', '$\sigma$=5', '$\sigma$=10', '$\sigma$-combined']
    metadata = {
        #'title':'{} test perf of sigma-combined net'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epoch',
        'ylabel': 'test acc',
        'label': label_list
    }
    plot.plot_epoch_log(input_list, 'test_acc', metadata)
    filepath = get_exp_path("fig6b-{}-sigmacomb-test-acc.{}".format(get_plot_filename(config_dict), config_dict['ext']))
    save_experiment_fig(filepath)

def task_sigmacomb_func(config_dict):

    model_filename = config_dict['model_file']

    if isinstance(config_dict['exp_index'], list):
        model_list = ["{}-{:04d}.pt".format(os.path.splitext(model_filename)[0], i) for i in config_dict['exp_index']]
        config_dict['model_file'] = model_list
        config_dict['exp_index'] = len(config_dict['exp_index']) + 1
        training_format_combined(config_dict)
    else:
        func = task_assigner(config_dict['training_type'])
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        func(config_dict)


