from .. import *
from .  import *

def activations_extraction(model, data_loader, out_dim=10, hid_idx=-1,):

    out_activation = np.zeros([len(data_loader)*data_loader.batch_size, out_dim])
    out_label = np.zeros([len(data_loader)*data_loader.batch_size,])
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(data_loader):
        
        if len(data)<data_loader.batch_size:
            break

        data = data.to(device)
        output, hiddens = model(data)
        
        begin = batch_idx*data_loader.batch_size
        end = (batch_idx+1)*data_loader.batch_size
        out_activation[begin:end] = hiddens[hid_idx].detach().cpu().numpy()
        out_label[begin:end] = target.detach().cpu().numpy()
        
    return {"activation":out_activation, "label":out_label}


def hsic_objective(hidden, h_target, h_data, sigma):


    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)


    return hsic_hx_val, hsic_hy_val
  
def model_distribution(config_dict):

    if config_dict['model'] == 'needle':
        model = ModelNeedle(**config_dict)
    elif config_dict['model'] == 'conv':
        model = ModelConv(**config_dict)
    elif config_dict['model'] == 'linear':
        model = ModelLinear(**config_dict)
    elif config_dict['model'] == 'resnet-linear':
        model = ModelResLinear(**config_dict)
    elif config_dict['model'] == 'resnet-conv':
        model = ModelResConv(**config_dict)
    else:
        raise ValueError("Unknown model name or not support [{}]".format(config_dict['model']))

    return model
