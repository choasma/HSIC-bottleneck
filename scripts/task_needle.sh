#!/bin/bash

##### deprecated in current paper. 

export HSICBT_TIMESTAMP=`date +"%d%m%y_%H%M%S"`
#export HSICBT_TIMESTAMP=221119_131922
task=hsictrain # backprop/hsictrain
run_hsicbt -cfg config/needle.yaml -tt ${task} -ep 50
run_plot -t needle -dc mnist -e pdf -tt ${task} -fn ${HSICBT_TIMESTAMP}_needle-${task}-mnist.npy -e pdf

# task=backprop # backprop/hsictrain
# run_hsicbt -cfg config/needle.yaml -tt ${task} -ep 50
# run_plot -t needle -dc mnist -e pdf -tt ${task} -fn ${HSICBT_TIMESTAMP}_needle-${task}-mnist.npy -e pdf

# for i in {0..9}
# do
#     run_plot -t needle -dc mnist -e pdf -ttt backprop -fn ${HSICBT_TIMESTAMP}_needle-backprop-mnist.npy -id $i -e pdf
# done


