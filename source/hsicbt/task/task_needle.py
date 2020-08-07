from . import *

def plot_needle_result(config_dict):
    try:

        out = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code']))['batch_log_list']

        if config_dict['plot_task'] == 1:
            if config_dict.get('filepath'):
                config_dict['exp_idx'] = ''
                filepath = get_act_raw_path_(os.path.basename(config_dict['filepath']), config_dict['exp_idx'])
                fig = plot.plot_1d_activation_kde(filepath)
            else:
                fig = plot.plot_1d_activation_kde(get_act_path(config_dict['task'], config_dict['training_type'], config_dict['data_code']))

            #config_dict['footnote'] += "; nHSIC(X,Z_L)={:.2f} nHSIC(Y,Z_L)={:.2f}".format(
            #    np.mean(out[int(config_dict['exp_idx'])]['batch_hsic_hx']),
            #    np.mean(out[int(config_dict['exp_idx'])]['batch_hsic_hy']),                
            #)
            plot.adding_footnote(fig, config_dict['footnote'])

            if config_dict['training_type'] == TTYPE_HSICTRAIN:
                filepath = get_experiment_fig_filename(config_dict, 'needle-1', config_dict['exp_idx'])
            elif config_dict['training_type'] == TTYPE_STANDARD:
                filepath = get_experiment_fig_filename(config_dict, 'needle-2', config_dict['exp_idx'])
            save_experiment_fig(filepath)

        # input_list = [out]
        # label_list = ['val']
        # metadata = {
        #     #'title':'nHSIC(X, Z_L) of Varied-activation',
        #     'title': '',
        #     'xlabel': '',
        #     'ylabel': 'nHSIC(X, Z_L)',
        #     'label': label_list
        # }
        # plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
        # filepath = get_exp_path("test_hsicxz.{}".format(config_dict['ext']))
        # save_experiment_fig(filepath)


        # out = load_logs(get_log_filepath(
        #     config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code']))['batch_log_list']
        # input_list = [out]
        # label_list = ['val']
        # metadata = {
        #     #'title':'nHSIC(X, Z_L) of Varied-activation',
        #     'title': '',
        #     'xlabel': '',
        #     'ylabel': 'nHSIC(Y, Z_L)',
        #     'label': label_list
        # }
        # plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
        # filepath = get_exp_path("test_hsicyz.{}".format(config_dict['ext']))
        # save_experiment_fig(filepath)
        
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()    

def task_needle_func(config_dict):

    func = task_assigner(config_dict['training_type'])
    func(config_dict)


