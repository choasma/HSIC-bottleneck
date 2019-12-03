#!/bin/bash

run_hsicbt -cfg config/resconv-hsicbt.yaml   -mf hsic_weight_resconv_mnist.pt -dc mnist
run_hsicbt -cfg config/resconv-format.yaml   -mf hsic_weight_resconv_mnist.pt -dc mnist
run_hsicbt -cfg config/resconv-backprop.yaml -mf hsic_weight_resconv_mnist.pt -dc mnist
run_hsicbt -cfg config/resconv-hsicbt.yaml   -mf hsic_weight_resconv_fmnist.pt -dc fmnist -s 5 -ld 500
run_hsicbt -cfg config/resconv-format.yaml   -mf hsic_weight_resconv_fmnist.pt -dc fmnist
run_hsicbt -cfg config/resconv-backprop.yaml -mf hsic_weight_resconv_fmnist.pt -dc fmnist
run_hsicbt -cfg config/resconv-hsicbt.yaml   -mf hsic_weight_resconv_cifar10.pt -dc cifar10 -s 5 -ld 500
run_hsicbt -cfg config/resconv-format.yaml   -mf hsic_weight_resconv_cifar10.pt -dc cifar10 
run_hsicbt -cfg config/resconv-backprop.yaml -mf hsic_weight_resconv_cifar10.pt -dc cifar10 
run_plot -t resconv -e pdf

