from .. import *
from .block import *
from ..utils.misc import *

class ModelResConv(nn.Module):

    def __init__(self, in_width=784, hidden_width=64, n_layers=5, atype='relu', 
        last_hidden_width=None, model_type='simple-dense', data_code='mnist', **kwargs):
        super(ModelResConv, self).__init__()
    
        block_list = []
        is_conv = False

        in_ch = get_in_channels(data_code)

        last_hw = hidden_width

        if last_hidden_width:
            last_hw = last_hidden_width
        
        for i in range(n_layers):
            block = BasicResidualBlockConv(hidden_width, hidden_width, atype)
            block_list.append(block)

        self.input_layer    = makeblock_conv(in_ch, hidden_width, atype, stride=2)
        self.sequence_layer = nn.Sequential(*block_list)
        self.pooling = nn.MaxPool2d((2,2))

        if data_code == 'cifar10':
            factor = 196
        else:
            factor = 144

        self.output_layer   = makeblock_dense(factor*int(hidden_width), last_hidden_width, atype)

        self.is_conv = is_conv
        self.in_width = in_width

    def forward(self, x):

        output_list = []
        
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x)
            output_list.append(x)
        # x = self.pooling(x)
        
        x = x.view(-1, np.prod(x.size()[1:]))

        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list
