#!/bin/bash

# please refer to bin/run_hsicbt to see the tuneable parameters,
# for this tutorial, sigma and batch_size are tested
p_sigma=(50 100 150 200)
p_lambda=(1000) # beta, the hsic-bt equation on HSIC(Z_i,Y)
p_learningrate=(0.001)

# some other params with non-array style
p_batchsize=512
p_depth=5 # num of hidden layer
n_epoch=1 # num of hsic-training epoch
n_kernels=64 # num of kernels at each layer
t_data=cifar10


# # # grid on HSIC-training
for p_s in ${p_sigma[@]}
do
    for p_l in ${p_lambda[@]}
    do
        for p_lr in ${p_learningrate[@]}
        do
            echo -e "\e[1m\e[100m======>" \
                 "${t_data}," \
                 "sigma: ${p_s}," \
                 "lambda: ${p_l}," \
                 "batch_size: ${p_batchsize}," \
                 "learning rate: ${p_lr}," \
                 "n_epoch: ${n_epoch}," \
                 "n_layers: ${p_depth}," \
                 "n_kernels: ${n_kernels}" \
                 "\e[0m"
            
            export HSICBT_TIMESTAMP=`date +"%d%m%y_%H%M%S"` 
            # # # for grid-searching resnet (HSIC-training first then format training), adding -vb for verbosing the network architecture
            run_hsicbt -cfg config/resconv-hsicbt.yaml -mf hsic_weight_resconv_${t_data}.pt \
                       -tt hsictrain -ep ${n_epoch} -s ${p_s} -ld ${p_l} -lr ${p_lr} \
                       -dc ${t_data} -bs ${p_batchsize} -dt ${p_depth} -d ${n_kernels}

            # # # and we evalute at format-training only 1 epoch, adding -vb for verbosing the network architecture
            run_hsicbt -cfg config/resconv-format.yaml -mf hsic_weight_resconv_${t_data}.pt -dc ${t_data} \
                       -tt format -ep 1 -s ${p_s} -ld ${p_l} -lr 0.1 -dc ${t_data} -bs ${p_batchsize} -dt ${p_depth} -d ${n_kernels}

        done
    done
done



# # # grid-on backprop training only with learning rate
# p_learningrate=(0.1)
# for p_lr in ${p_learningrate[@]}
# do
#     run_hsicbt -cfg config/resconv-backprop.yaml -mf hsic_weight_resconv_${t_data}.pt -dc ${t_data} \
    #                -tt backprop -ep 1 -lr ${p_learningrate} -dc ${t_data} -bs ${p_batchsize} -dt ${p_depth} -d ${n_kernels}
# done


# # # if you want to evaluate format-training with different resnet model saved at each epoch
# for i in  $(seq -f "%04g" 1 49)
# do
#     echo ${i}
#     run_hsicbt -cfg config/resconv-format.yaml -mf raw/161019_185352_hsic_weight_resconv_${t_data}-${i}.pt -dc ${t_data} \
    #                -tt format -ep 3 -s 1 -ld 1 -lr 0.1 -dc ${t_data} -bs ${p_batchsize} -dt ${p_depth}
# done
