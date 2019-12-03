from .. import *
from .const import *
from .color import *
from .misc  import *
from .path  import *
from .const import *
import yaml

def load_yaml(filepath):

    with open(filepath, 'r') as stream:
        try:
            data = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    print_highlight("Loaded  [{}]".format(filepath))
    return data
    
def save_model(model, filepath, sympath):
    timestamp_path = attaching_timestamp_filepath(filepath)
    torch.save(model.state_dict(), timestamp_path)
    print_highlight("Saved   [{}]".format(timestamp_path))
    make_symlink(timestamp_path, sympath)
    
def load_model(filepath):
    model = torch.load(filepath)
    print_highlight("Loaded  [{}]".format(filepath))
    return model

def save_logs(logs, filepath):
    timestamp_path = attaching_timestamp_filepath(filepath)
    np.save(timestamp_path, logs)
    make_symlink(timestamp_path, filepath)
    print_highlight("Saved   [{}]".format(timestamp_path), ctype="blue")

def load_logs(filepath):
    logs = np.load(filepath, allow_pickle=True)[()]
    print_highlight("Loaded  [{}]".format(filepath), ctype="blue")
    return logs

