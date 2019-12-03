import os
from .path import *
from .const import *
from .color import *
import glob

def attaching_timestamp_filepath(filepath):
    filename = os.path.basename(filepath)
    dirname  = os.path.dirname(filepath)
    filename, ext = os.path.splitext(filename)
    filename_time = "{}_{}{}".format(TIMESTAMP_CODE, filename, ext)
    timestamp_path = os.path.join(dirname, 'raw', filename_time)
    return timestamp_path

def make_symlink(src_path, sym_path):
    if os.path.exists(sym_path):
        os.remove(sym_path)
    os.symlink(src_path, sym_path)
    print_highlight("Symlink [{}]".format(sym_path), ctype="blue")

def code_name(task, ttype, dtype, idx):
    if idx:
        filename = "{}-{}-{}-{:04d}.npy".format(task, ttype, dtype, idx)
    else:
        filename = "{}-{}-{}.npy".format(task, ttype, dtype)
    return filename

def get_log_filepath(task, ttype, dtype, idx=None):
    filepath = "{}/assets/logs/{}".format(os.getcwd(), code_name(task, ttype, dtype, idx))
    return filepath

def get_log_raw_filepath(task, ttype, dtype, idx=None):
    filepath = "{}/assets/raw/logs/{}".format(os.getcwd(), code_name(task, ttype, dtype, idx))
    return filepath

#### debug
def get_log_filepath_(config_dict):
    return "{}/assets/logs/{}".format(os.getcwd(), filename)

#### debug
def get_log_raw_filepath_(filename):
    return "{}/assets/logs/raw/{}".format(os.getcwd(), filename)

def get_plot_filename(config_dict):
    return "{}-{}".format(config_dict['task'], config_dict['data_code'])

def get_exp_path(filename):
    return "{}/assets/exp/{}".format(os.getcwd(), filename)

def get_exp_raw_path(filename):
    return "{}/assets/exp/raw/{}".format(os.getcwd(), filename)

def get_act_path(task, ttype, dtype, idx=None):
    filepath = "{}/assets/activation/{}".format(os.getcwd(), code_name(task, ttype, dtype, idx))
    return filepath

def get_act_raw_path(task, ttype, dtype, idx=None):
    filepath = "{}/assets/activation/raw/{}".format(os.getcwd(), code_name(task, ttype, dtype, idx))
    return filepath

def get_act_path_(task, ttype, dtype, idx=None):
    filepath = "{}/assets/activation/{}".format(os.getcwd(), code_name(task, ttype, dtype, idx))
    return filepath

def get_act_raw_path_(filename, idx=''):
    if idx:
        idx = "-{:04d}".format(idx)

    filepath = "{}/assets/activation/raw/{}{}.{}".format(os.getcwd(), filename[:-4], idx, filename[-3:])
    return filepath

def get_model_path(filename, idx=None):
    if idx:
        filepath = "{}/assets/models/{}-{:04d}.pt".format(
            os.getcwd(), os.path.splitext(filename)[0], idx)
    else:
        filepath = "{}/assets/models/{}".format(os.getcwd(), filename)
    return filepath

def get_model_raw_path(filename, idx=None):
    if idx:
        filepath = "{}/assets/models/raw/{}-{:04d}.pt".format(
            os.getcwd(), os.path.splitext(filename)[0], idx)
    else:
        filepath = "{}/assets/models/raw/{}".format(os.getcwd(), filename)
    return filepath

def get_tmp_path(filename):
    return "{}/assets/tmp/{}".format(os.getcwd(), filename)


