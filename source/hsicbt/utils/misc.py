from .. import *

def get_current_timestamp():
    return strftime("%y%m%d_%H%M%S")

def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'cifar10':
        in_ch = 3
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

def get_in_dimensions(data_code):
    in_dim = -1    
    if data_code == 'mnist':
        in_dim = 784
    elif data_code == 'cifar10':
        in_dim = 1024
    elif data_code == 'fmnist':
        in_dim = 784
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_dim

def get_accuracy_epoch(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    acc = []
    loss = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = model.to('cuda')
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        output, hiddens = model(data)
        loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
        acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
    return np.mean(acc), np.mean(loss)


def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_accuracy_hsic(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    for batch_idx, (data, target) in enumerate(dataloader):
        output, hiddens = model(data.to(next(model.parameters()).device))
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy().reshape(-1,1)
        output_list.append(output)
        target_list.append(target)
    output_arr = np.vstack(output_list)
    target_arr = np.vstack(target_list)
    avg_acc = 0
    reorder_list = []
    for i in range(10):
        indices = np.where(target_arr==i)[0]
        select_item = output_arr[indices]
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        while np.argmax(y) in reorder_list:
            y[np.argmax(y)] = 0
        reorder_list.append(np.argmax(y))
        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0])/float(out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.

    return avg_acc*100., reorder_list

def get_layer_parameters(model, idx_range):

    param_out = []
    param_out_name = []
    for it, (name, param) in enumerate(model.named_parameters()):
        if it in idx_range:
            param_out.append(param)
            param_out_name.append(name)

    return param_out, param_out_name


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y])
