from .. import *


def get_activation(atype):

    if atype=='relu':
        nonlinear = nn.ReLU()
    elif atype=='tanh':
        nonlinear = nn.Tanh() 
    elif atype=='sigmoid':
        nonlinear = nn.Sigmoid() 
    elif atype=='elu':
        nonlinear = nn.ELU()

    return nonlinear

def get_activation_functional(atype):

    if atype=='relu':
        nonlinear = torch.relu
    elif atype=='tanh':
        nonlinear = torch.tanh
    elif atype=='sigmoid':
        nonlinear = torch.sigmoid
    elif atype=='elu':
        nonlinear = torch.elu

    return nonlinear

def get_primative_block(model_type, hid_in, hid_out, atype):
    if model_type=='simple-dense':
        block = makeblock_dense(hid_in, hid_out, atype)
    elif model_type=='simple-conv':
        block = makeblock_conv(hid_in, hid_out, atype)
    elif model_type=='resnet-dense':
        block = BasicResidualBlockDense(hid_in, hid_out, atype)
    elif model_type=='resnet-conv':
        block = BasicResidualBlockConv(hid_in, hid_out, atype)
    return block

def makeblock_dense(in_dim, out_dim, atype):
    
    layer = nn.Linear(in_dim, out_dim)
    bn = nn.BatchNorm1d(out_dim, affine=False)
    if atype=='linear':
        out = nn.Sequential(*[layer, bn])
    else:
        nonlinear = get_activation(atype)
        out = nn.Sequential(*[layer, bn, nonlinear])
    return out

def makeblock_conv(in_chs, out_chs, atype, stride=1):

    layer = nn.Conv2d(in_channels=in_chs, 
        out_channels=out_chs, kernel_size=5, stride=stride)
    bn = nn.BatchNorm2d(out_chs, affine=False)
    nonlinear = get_activation(atype)

    return nn.Sequential(*[layer, bn, nonlinear])

class BasicBlockConv(nn.Module):
    """docstring for BasicBlockConv"""
    def __init__(self, in_chs, out_chs, atype):
        super(BasicBlockConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_chs, 
            out_channels=out_chs, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_chs, affine=False)
        self.nfunc = get_activation_functional(atype)

    def forward(self, x):        
        out = self.nfunc(self.bn(self.conv(x)))
        return out
       
class BasicBlockDense(nn.Module):
    """docstring for BasicBlockDense"""
    def __init__(self, in_dim, out_dim, atype):
        super(BasicBlockDense, self).__init__()

        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.nfunc = get_activation_functional(atype)

    def forward(self, x):        
        out = self.nfunc(self.bn(self.dense(x)))
        return out

class BasicResidualBlockDense(nn.Module):

    def __init__(self, in_dim, out_dim, atype):
        super(BasicResidualBlockDense, self).__init__()

        self.dense1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim, affine=False)
        self.shortcut = nn.Sequential()
        self.nfunc = get_activation_functional(atype)
        self.bn3 = nn.BatchNorm1d(out_dime, affine=False)

    def forward(self, x):
        out = self.nfunc(self.bn1(self.dense1(x)))
        out += self.shortcut(x)
        out = self.bn3(out)
        out = self.nfunc(out)
        return out

class BasicResidualBlockConv(nn.Module):

    def __init__(self, in_chs, out_chs, atype):
        super(BasicResidualBlockConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(out_chs, affine=False)
        self.shortcut = nn.Sequential()
        self.nfunc = get_activation_functional(atype)
        self.bn3 = nn.BatchNorm2d(out_chs, affine=False)

    def forward(self, x):

        out = self.nfunc(self.bn1(self.conv1(x)))
        out += self.shortcut(x)
        out = self.bn3(out)
        out = self.nfunc(out)

        return out

