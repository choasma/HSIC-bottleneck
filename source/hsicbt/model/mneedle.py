from .. import *
from .block import *
from ..utils.misc import *

class ModelNeedle(nn.Module):

    def __init__(self, data_code='mnist', atype='tanh', **kwargs):
        super(ModelNeedle, self).__init__()
    
        in_dim = get_in_dimensions(data_code)
        in_ch = get_in_channels(data_code)
        self._in_width = in_dim*in_ch
        
        self.f1  = nn.Linear(self._in_width, 256)
        self.b1  = nn.BatchNorm1d(256)
        self.f2  = nn.Linear(256, 128)
        self.b2  = nn.BatchNorm1d(128)
        self.f3  = nn.Linear(128, 64)
        self.b3  = nn.BatchNorm1d(64)
        self.f4  = nn.Linear(64, 32)
        self.b4  = nn.BatchNorm1d(32)
        self.f5  = nn.Linear(32, 16)
        self.b5  = nn.BatchNorm1d(16)
        self.f6  = nn.Linear(16, 8)
        self.b6  = nn.BatchNorm1d(8)
        self.f7  = nn.Linear(8, 1)
        self.b7  = nn.BatchNorm1d(1)
        self.out = nn.Linear(1, 10)

        self._f = get_activation_functional(atype)


    def forward(self, x):

        output_list = []
        x = x.view(-1, self._in_width)
        
        x = self.b1(self.f1(x))
        x = self._f(x)
        output_list.append(x)
        x = self.b2(self.f2(x))
        x = self._f(x)
        output_list.append(x)
        x = self.b3(self.f3(x))
        x = self._f(x)
        output_list.append(x)
        x = self.b4(self.f4(x))
        x = self._f(x)
        output_list.append(x)
        x = self.b5(self.f5(x))
        x = self._f(x)
        output_list.append(x)
        x = self.b6(self.f6(x))
        x = self._f(x)
        output_list.append(x)
        x = self.b7(self.f7(x))
        #x = F.hardtanh(x,-2.,2.)
        x = torch.tanh(x)
        output_list.append(x)
        x = self.out(x)

        return F.log_softmax(x, dim=1), output_list
