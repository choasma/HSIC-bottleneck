from .. import *
from .  import *
from .train_misc     import *
from ..utils.const   import *

batch_acc    = meter.AverageMeter()
batch_loss   = meter.AverageMeter()
batch_hischx = meter.AverageMeter()
batch_hischy = meter.AverageMeter()

def standard_train(cepoch, model, data_loader, optimizer, config_dict):

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    prec1 = total_loss = hx_l = hy_l = -1

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])

    n_data = config_dict['batch_size'] * len(data_loader)
    
    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=150)
    # for batch_idx, (data, target) in enumerate(data_loader):
    for batch_idx, (data, target) in pbar:

        if os.environ.get('HSICBT_DEBUG')=='4':
            if batch_idx > 5:
                break

        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])
        output, hiddens = model(data)

        h_target = target.view(-1,1)
        h_target = misc.to_categorical(h_target, num_classes=10).float()
        
        h_data = data.view(-1, np.prod(data.size()[1:]))

        # # # if want to monitor hsic
        if config_dict['task'] == 'varied-activation' or config_dict['task'] == 'varied-depth':
            hx_l, hy_l = hsic_objective(
                    hiddens[-1],
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=config_dict['sigma']
                )
            hx_l = hx_l.cpu().detach().numpy()
            hy_l = hy_l.cpu().detach().numpy()

        optimizer.zero_grad()
        loss = cross_entropy_loss(output, target)
        loss.backward()
        optimizer.step()


        loss = float(loss.detach().cpu().numpy())
        prec1, prec5 = misc.get_accuracy(output, target, topk=(1, 5)) 
        prec1 = float(prec1.cpu().numpy())
    
        batch_acc.update(prec1)   
        batch_loss.update(loss)  
        batch_hischx.update(hx_l)
        batch_hischy.update(hy_l)

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.4f} hsic_yz:{hsic_zy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        loss = batch_loss.avg, 
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )

        # # # preparation log information and print progress # # #
        if ((batch_idx) % config_dict['log_batch_interval'] == 0): 
            batch_log['batch_acc'].append(batch_acc.val)
            batch_log['batch_loss'].append(batch_loss.val)
            batch_log['batch_hsic_hx'].append(batch_hischx.val)
            batch_log['batch_hsic_hy'].append(batch_hischy.val)

        pbar.set_description(msg)

    return batch_log
