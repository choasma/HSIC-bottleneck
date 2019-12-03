#!/bin/bash

# """
# this is the script that produce the results all at once without training
# """

# result file format
ftype=pdf

# debug training (from hsic-train, ba# ckprop to format train)
run_plot -t general -dc mnist -e $ftype

# # fig2a-c: hsic monitoring in backprop training with vaired activation functions
run_plot -t varied-activation -dc mnist -e $ftype

# # fig2d-f: hsic monitoring in backprop training with varied network depth
run_plot -t varied-depth -dc mnist -e $ftype

# # fig4
run_plot -t hsic-solve -dc mnist -e $ftype
run_plot -t hsic-solve -dc fmnist -e $ftype
run_plot -t hsic-solve -dc cifar10 -e $ftype

# # fig5: format-training with different hsic-trained network
run_plot -t varied-epoch -dc mnist -e $ftype

# # fig6-a: hsic-trained network capacity with network size
run_plot -t varied-dim -dc mnist -e $ftype

# # fig6-b: hsic-trained network capacity with combo
run_plot -t sigma-combined -dc mnist -e $ftype

# # fig7: resnet case
run_plot -t resconv -e $ftype


