from .. import *

class ModelEnsembleComb(nn.Module):

    def __init__(self, hsic_models, vanilla_model):
        super(ModelEnsembleComb, self).__init__()
        # self.intput_model = intput_model
        self._hsic_models = hsic_models
        self._vanilla_model = vanilla_model
        
    def forward(self, x):

        x_list = []
        for hsic_model in self._hsic_models:
            h, hiddens = hsic_model(x)
            x_list.append(torch.unsqueeze(h, dim=0))
    
        x = torch.mean(torch.cat(x_list), dim=0)        
        bn = nn.BatchNorm1d(x.size()[1]).to('cuda')
        x = bn(x)
        x = self._vanilla_model(x)
        
        return F.log_softmax(x, dim=1), hiddens
