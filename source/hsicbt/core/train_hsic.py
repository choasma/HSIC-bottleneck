from .. import *
from .  import *
from .train_misc     import *
from ..utils.const   import *

def hsic_train(cepoch, model, data_loader, config_dict):

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    prec1 = total_loss = hx_l = hy_l = -1

    batch_acc    = meter.AverageMeter()
    batch_loss   = meter.AverageMeter()
    batch_hischx = meter.AverageMeter()
    batch_hischy = meter.AverageMeter()

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])

    n_data = config_dict['batch_size'] * len(data_loader)

    # sigma_optimizer = optim.SGD([sigma_tensor], lr=1E-5)

    # for batch_idx, (data, target) in enumerate(data_loader):
    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=120)
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

     
        idx_range = []
        it = 0

        # So the batchnorm is not learnable, making only @,b at layer
        for i in range(len(hiddens)):
            idx_range.append(np.arange(it, it+2).tolist())
            it += 2
    
        for i in range(len(hiddens)):
            
            output, hiddens = model(data)
            params, param_names = misc.get_layer_parameters(model=model, idx_range=idx_range[i]) # so we only optimize one layer at a time
            optimizer = optim.SGD(params, lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)
            optimizer.zero_grad()
            # sigma_optimizer.zero_grad()
            if len(hiddens[i].size()) > 2:
                hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))

            # WIP: special case for needle
            if config_dict['task']=='needle' and i==len(hiddens)-1:
                h_target = target.view(-1,1).float()
                #h_target =  h_target/9.


            hx_l, hy_l = hsic_objective(
                    hiddens[i],
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=config_dict['sigma'],
            )
            #print(torch.max(hiddens[i]).cpu().detach().numpy(), torch.min(hiddens[i]).cpu().detach().numpy(), torch.std(hiddens[i]).cpu().detach().numpy())
            loss = hx_l - config_dict['lambda_y']*hy_l
            loss.backward()
            optimizer.step()
            # sigma_optimizer.step()
        # if config_dict['hsic_solve']:
        #     prec1, reorder_list = misc.get_accuracy_hsic(model, data_loader)
        batch_acc.update(prec1)
        batch_loss.update(total_loss)
        batch_hischx.update(hx_l.cpu().detach().numpy())
        batch_hischy.update(hy_l.cpu().detach().numpy())

        # # # preparation log information and print progress # # #

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] H_hx:{H_hx:.4f} H_hy:{H_hy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        H_hx = batch_hischx.avg, 
                        H_hy = batch_hischy.avg,
                )

        if ((batch_idx+1) % config_dict['log_batch_interval'] == 0):

            batch_log['batch_acc'].append(batch_loss.avg)
            batch_log['batch_loss'].append(batch_acc.avg)
            batch_log['batch_hsic_hx'].append(batch_hischx.avg)
            batch_log['batch_hsic_hy'].append(batch_hischy.avg)

        pbar.set_description(msg)

        # if cepoch==1:
        #     data = activations_extraction(model, data_loader)
        #     _code_name = [config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code']+"_batch", batch_idx]
        #     filepath = get_act_path(*_code_name)
        #     save_logs(data, filepath)

    return batch_log
