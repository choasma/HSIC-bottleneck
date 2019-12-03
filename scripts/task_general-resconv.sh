#!/bin/bash


####### legacy code #######





# --cifar10/mnist/fashionmnist--
# task_general-resconv.sh [-d] datacode [-e] hsic-training epochs [-t] is hsic-training (usually it takes time)
#
# task_general-resconv.sh -d cifar10 -e 50 -t 1
# task_general-resconv.sh -d mnist   -e 30 -t 1
# task_general-resconv.sh -d fmnist  -e 50 -t 1



# https://www.lifewire.com/pass-arguments-to-bash-script-2200571
while getopts d:e:t: option
do
case "${option}"
in
d) datacode=${OPTARG};;
e) epochs=${OPTARG};;
t) istrainhsic=${OPTARG};;
esac
done

echo ${datacode} ${epochs} ${istrainhsic}

if [ $istrainhsic -eq 1 ]
then
   run_hsicbt -cfg config/general-hsicbt.yaml -tt hsictrain -m resnet-conv -dc ${datacode} -d 15 -ep ${epochs} -lr 0.0001 -lhw 512 -mf hsic_weight_resnetc_${datacode}.pt -vb
fi

# fmnist
run_hsicbt -cfg config/general-format.yaml   -tt format    -m resnet-conv -dc ${datacode} -d 15 -ep 5 -lr 0.0025 -lhw 512 -mf hsic_weight_resnetc_${datacode}.pt 
run_hsicbt -cfg config/general-backprop.yaml -tt backprop  -m resnet-conv -dc ${datacode} -d 15 -ep 5 -lr 0.0025 -lhw 512
run_plot -t general -dc ${datacode} -e pdf
