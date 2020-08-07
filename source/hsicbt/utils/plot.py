import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode']=True

import matplotlib.pyplot as plt
import numpy as np

from .color import *
from .const import *
from .misc import *
from .path import *

matplotlib.use('Agg')

def plot_epoch_log(curve_list, ptype, metadata):


    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    ax = fig.add_subplot(111)

    n = len(curve_list[0][ptype])
    xticks_idx = np.arange(n).tolist()
    xticks_val = np.arange(n).tolist()

    if n>10:
        skip = n/10
        xticks_idx = xticks_idx[::skip] + [xticks_idx[-1]]
        xticks_val = xticks_idx
    
    max_y = -1E5
    min_y = 1E5
    end_max_y = -1E5
    end_min_y = 1E5
    
    for i, curve_dict in enumerate(curve_list):
        p = ax.plot(curve_dict[ptype], linewidth=4, label=metadata['label'][i])
        ax.plot(curve_dict[ptype], '*', ms=30, alpha=.3, color=p[0].get_color())
        max_y = max(max_y, np.max(curve_dict[ptype]))
        min_y = min(min_y, np.min(curve_dict[ptype]))
        end_max_y = max(end_max_y, np.max(curve_dict[ptype][-1]))
        end_min_y = min(end_min_y, np.min(curve_dict[ptype][-1]))
    # WIP: adjusting legend dynamically
    # if 'acc' in ptype:
    #     if max_y < 50.:
    #         plt.legend(fontsize=FONTSIZE_LEDEND, loc='upper right')
    #     else:
    #         plt.legend(fontsize=FONTSIZE_LEDEND, loc='lower right')
    # elif 'loss' in ptype:
    #     plt.legend(fontsize=FONTSIZE_LEDEND, loc='upper right')
    # else:
    #     plt.legend(fontsize=FONTSIZE_LEDEND)

    if abs(end_max_y - end_min_y) > abs(max_y - min_y)/2.:
        plt.legend(fontsize=FONTSIZE_LEDEND)
    else:
        if curve_list[0][ptype][-1] > curve_list[0][ptype][0]:
            plt.legend(fontsize=FONTSIZE_LEDEND, loc='lower right')
        else:
            plt.legend(fontsize=FONTSIZE_LEDEND, loc='upper right')
    #plt.legend(fontsize=FONTSIZE_LEDEND)
        
    margin = abs(max_y - min_y)/2.
    max_y += margin
    min_y -= margin
    if 'acc' in ptype:
        yticks_idx = np.arange(110)[::10] # [0, 10, ..., 100]
        yticks_val = np.arange(110)[::10] # [0, 10, ..., 100]
    else:
        yticks_idx = np.linspace(min_y, max_y, 100)[::10]
        yticks_val = [np.round(x, 1) for x in np.linspace(min_y, max_y, 100)[::10]]
        
    ax.set_title(metadata['title'], fontsize=FONTSIZE_TITLE)
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_val, fontsize=FONTSIZE_LEDEND)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_val,  fontsize=FONTSIZE_YTICKS)
    ax.set_xlabel(metadata['xlabel'], fontsize=FONTSIZE_XLABEL)
    ax.set_ylabel(metadata['ylabel'], fontsize=FONTSIZE_YLABEL)


def plot_batches_log(curve_list, ptype, metadata):

    #assert len(curve_list)>1, "this is for multiple curve plotting"

    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    ax = fig.add_subplot(111)


    n = len(curve_list[0][0][ptype])
    if n==0:
        n=1
    xticks_idx = np.arange(0, n*(len(curve_list[0])+1), n).tolist()
    xticks_val = np.arange(len(xticks_idx)).tolist()

    n = len(xticks_idx)
    if n>10:
        skip = int(n/10)
        xticks_idx = xticks_idx[::skip] + [xticks_idx[-1]]
        xticks_val = xticks_val[::skip] + [xticks_val[-1]]
    
    max_y = -1E5
    min_y = 1E5
    end_max_y = -1E5
    end_min_y = 1E5
    start_max_y = -1E5
    start_min_y = 1E5
    
    for i, curve_dict in enumerate(curve_list):

        val = [x[ptype] for x in curve_dict]
        val = [y for x in val for y in x]


        e_w = 5
        e = np.zeros_like(val)
        v = np.zeros_like(val)
        for j in range(len(val)):
            i_b = np.max([0, j-e_w])
            i_e = np.min([len(val), j+e_w])
            e[j] = np.std(val[i_b:i_e])
            v[j] = np.mean(val[i_b:i_e])
        p = ax.plot(v, linewidth=4, label=metadata['label'][i])

        ax.fill_between(np.arange(len(val)), v-e, v+e, color=p[0].get_color(), alpha=.25)
        max_y = max(max_y, np.max(val))
        min_y = min(min_y, np.min(val))
        end_max_y = max(end_max_y, np.max(val[-1]))
        end_min_y = min(end_min_y, np.min(val[-1]))
        start_max_y = max(start_max_y, np.max(val[0]))
        start_min_y = min(start_min_y, np.min(val[0]))
                
        
    plt.legend(fontsize=FONTSIZE_LEDEND)

    margin = abs(max_y-min_y)/2.
    if (end_max_y+end_min_y) < (start_min_y+start_max_y):
        max_y += margin
    else:
        min_y -= margin
        
    if 'acc' in ptype:
        yticks_idx = np.arange(110)[::10] # [0, 10, ..., 100]
        yticks_val = np.arange(110)[::10] # [0, 10, ..., 100]
    else:
        yticks_idx = np.linspace(min_y, max_y, 100)[::10]
        yticks_val = [np.round(x,1) for x in np.linspace(min_y, max_y, 100)[::10]]

    ax.set_title(metadata['title'], fontsize=FONTSIZE_TITLE)
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_val, fontsize=FONTSIZE_XTICKS)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_val,  fontsize=FONTSIZE_YTICKS)
    ax.set_xlabel(metadata['xlabel'], fontsize=FONTSIZE_XLABEL)
    ax.set_ylabel(metadata['ylabel'], fontsize=FONTSIZE_YLABEL)


    val = [x[ptype] for x in curve_list[0]]
    if val[-1] > val[0]:
        plt.legend(fontsize=FONTSIZE_LEDEND, loc='lower right')
    else:
        plt.legend(fontsize=FONTSIZE_LEDEND, loc='upper right')
    

def plot_activation_distribution(datapath, title):


    data = np.load(datapath, allow_pickle=True)[()]
    activation_data = data['activation']
    label_data = data['label']
    label_index = []

    # # # calc average acc
    avg_acc = 0
    for i in range(10):
        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # get all associated classed activation
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        num_correct = np.where(out==np.argmax(y))[0] # comparing how many samples match the maximum of mean
        accuracy = float(num_correct.shape[0]/out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.
    ylim_min = np.min(activation_data)
    ylim_max = np.max(activation_data)

    fig, ax = plt.subplots(2, 5, figsize=(30,10))
    ax = [x for a in ax for x in a]
    
    shuffled_list = []
    for i in range(10):

        subplot = ax[i]

        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # extract activation associated with label
        out = np.array([np.argmax(vec) for vec in select_item]) # find the maximum arg of activation dist

        y = np.mean(select_item, axis=0)

        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0]/out.shape[0])

        e = np.std(select_item, axis=0)
        idx = np.arange(10).tolist()
        shuffled_list.append(int(np.argmax(y)))
        subplot.plot(y)
        subplot.fill_between(np.arange(10), y-e, y+e, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        #subplot.set_title("img class/argmax: {}/{}; category acc {:.2f}".format(i,int(np.argmax(y)), accuracy))
        subplot.set_xticks(idx)
        subplot.set_xticklabels(idx, fontsize=FONTSIZE_XTICKS)
        subplot.set_yticks(range(-1,4))
        subplot.set_yticklabels(range(-1,4), fontsize=FONTSIZE_YTICKS)
        subplot.set_ylim(-1, 4)
        #subplot.set_xlabel('10 dimension output activation indices')
        #subplot.set_ylabel('activation value')

    # fig.suptitle("activations of unformatted training", fontsize=FONTSIZE_TITLE)
    title += "  class argmax {}".format(shuffled_list)
    fig.suptitle(title, fontsize=FONTSIZE_TITLE)
    #fig.text(0.2, 0.05, "10 dimension output activation indices; shuffled argmax list {}".format(shuffled_list), fontsize=FONTSIZE_XLABEL)
    print(shuffled_list)
    #fig.text(0.1, 0.6, "activation", fontsize=FONTSIZE_YLABEL, rotation='vertical')

def plot_1d_activation_kde(datapath):

    data = np.load(datapath, allow_pickle=True)[()]
    activation_data = data['activation']
    label_data = data['label']

    from scipy.stats import gaussian_kde

    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    ax = fig.add_subplot(111)

    sample_idx = np.linspace(-1.3,1.3,150)
    for i in range(10):
        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # get all associated classed activation

        kernel = gaussian_kde(np.squeeze(select_item))

        sampling = kernel(sample_idx)

        plt.plot(sampling, linewidth=3, label="c:{} m:{:.2f}".format(i, float(np.mean(select_item))))


    xticks_idx = np.arange(len(sample_idx))
    xticks_idx = list(xticks_idx[::25]) + [xticks_idx[-1]]
    xticks_val = list(sample_idx[::25]) + [sample_idx[-1]]
    xticks_val = ["{:.1f}".format(x) for x in xticks_val]
   
    # plt.legend(fontsize=18) 
    #ax.set_title('class signals of dataset', fontsize=FONTSIZE_TITLE)
    ax.set_xticks(xticks_idx)
    #ax.set_ylim(0, 10)
    ax.set_xticklabels(xticks_val,  fontsize=FONTSIZE_XTICKS)
    ax.set_yticklabels(np.arange(10),  fontsize=FONTSIZE_YTICKS)
    ax.set_xlabel('tanh activation', fontsize=FONTSIZE_XLABEL)
    ax.set_ylabel('KDE density', fontsize=FONTSIZE_YLABEL)

    return fig

def adding_footnote(fig, text):
    fig.text(0., 0., text, fontsize=FONTSIZE_FOOTNOTE)
    
def save_figure(filepath):
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print_highlight("Saved   [{}]".format(filepath), ctype='blue')
