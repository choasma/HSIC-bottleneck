#!/bin/bash

# please refer to bin/run_hsicbt to see the tuneable parameters,
# for this tutorial, sigma and batch_size are tested
#p_sigma=(1 2 3 4 5 10 15 20 40 50)
p_sigma=(5 6 7 8)
p_batchsize=(256)
p_lambda=(100)
p_learningrate=(0.0005)
p_depth=5
n_epoch=1
t_type=hsictrain # backprop/hsictrain
t_data=mnist

# unfortunately, the nested loopping is the current solution. will improve it later
for p_s in ${p_sigma[@]}
do
    for p_l in ${p_lambda[@]}
    do
        for p_lr in ${p_learningrate[@]}
        do
            echo -e "\e[1m\e[100m ======>" \
                 "${t_data};" \
                 "sigma: ${p_s}," \
                 "lambda: ${p_l}," \
                 "batch_size: ${p_batchsize}," \
                 "learning rate: ${p_lr}," \
                 "n_epoch: ${n_epoch}," \
                 "n_layers: ${p_depth}," \
                 "\e[0m"
            
            export HSICBT_TIMESTAMP=`date +"%d%m%y_%H%M%S"`

            # # # for grid-searching unformat-training
            run_hsicbt -cfg config/general-hsicbt.yaml \
                       -tt ${t_type} -ep ${n_epoch} -s ${p_s} -ld ${p_l} -lr ${p_lr} -dc ${t_data} -bs ${p_batchsize} -dt ${p_depth}
            run_hsicbt -cfg config/general-format.yaml -ep 1
            
        done
    done
done

