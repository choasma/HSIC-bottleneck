from .. import *

class ModelEnsemble(nn.Module):

    def __init__(self, hsic_model, vanilla_model):
        super(ModelEnsemble, self).__init__()
        # self.intput_model = intput_model
        self._hsic_model = hsic_model
        self._vanilla_model = vanilla_model
        
    def forward(self, x):
        
        x, hiddens = self._hsic_model(x)
        x = self._vanilla_model(x)
        return x, hiddens