from hsicbt.core.train_hsic import hsic_train
from hsicbt.model.mhlinear import ModelLinear
from hsicbt.utils.dataset import get_dataset_from_code

# # # configuration
config_dict = {}
config_dict['batch_size'] = 128
config_dict['learning_rate'] = 0.001
config_dict['lambda_y'] = 100
config_dict['sigma'] = 2
config_dict['task'] = 'hsic-train'
config_dict['device'] = 'cuda'
config_dict['log_batch_interval'] = 10

# # # data prepreation
train_loader, test_loader = get_dataset_from_code('mnist', 128)

# # # simple fully-connected model
model = ModelLinear(hidden_width=64,
                    n_layers=3,
                    atype='relu',
                    last_hidden_width=None,
                    model_type='simple-dense',
                    data_code='mnist')

# # # start to train
epochs = 5
for cepoch in range(epochs):
    # you can also re-write hsic_train function
    hsic_train(cepoch, model, train_loader, config_dict)
