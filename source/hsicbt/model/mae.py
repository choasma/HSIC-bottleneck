from .. import *
from .block import *
from ..utils.misc import *

class ModelAutoEncoder(nn.Module):

    def __init__(self,  **kwargs):
        super(ModelAutoEncoder, self).__init__()
    
        block_list = []
        
        atype = 'relu'
        
        block = get_primative_block('simple-dense', 256, 128, atype)
        block_list.append(block)
        block = get_primative_block('simple-dense', 128, 64, atype)
        block_list.append(block)
        block = get_primative_block('simple-dense', 64, 128, atype)
        block_list.append(block)
        block = get_primative_block('simple-dense', 128, 256, atype)
        block_list.append(block)


        self.input_layer    = makeblock_dense(784, 256, atype)
        self.sequence_layer = nn.Sequential(*block_list)
        self.output_layer   = nn.Linear(256, 784)


    def forward(self, x):

        output_list = []
        x = x.view(-1, 784)
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x)
            output_list.append(x)
        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list
