#!/bin/bash

# test with 5,50 hidden layers
n_layers=5
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt backprop  -dc mnist -lr 0.05 -dt ${n_layers}
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc mnist -lr 0.001 -s 6 -ld 200 -dt ${n_layers}
run_plot -t hsic-solve -dc mnist -e pdf
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt backprop  -dc fmnist -lr 0.05 -dt ${n_layers}
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc fmnist -lr 0.001 -s 6 -ld 1000 -dt ${n_layers}
run_plot -t hsic-solve -dc fmnist -e pdf
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt backprop  -dc cifar10 -lr 0.05 -dt ${n_layers}
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc cifar10 -lr 0.001 -s 6 -ld 2500 -dt ${n_layers}
run_plot -t hsic-solve -dc cifar10 -e pdf
mv assets/exp/fig4-hsic-solve-mnist-test-acc.pdf assets/exp/fig4-layer5-hsic-solve-mnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-fmnist-test-acc.pdf assets/exp/fig4-layer5-hsic-solve-fmnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-cifar10-test-acc.pdf assets/exp/fig4-layer5-hsic-solve-cifar10-test-acc.pdf
mv assets/exp/fig3-hsic-solve-actdist-mnist.pdf assets/exp/fig3-layer5-hsic-solve-actdist-mnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-fmnist.pdf assets/exp/fig3-layer5-hsic-solve-actdist-fmnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-cifar10.pdf assets/exp/fig3-layer5-hsic-solve-actdist-cifar10.pdf

n_layers=50
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt backprop  -dc mnist -lr 0.01 -dt ${n_layers}
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc mnist -lr 0.001 -s 6 -ld 2500 -dt ${n_layers}
run_plot -t hsic-solve -dc mnist -e pdf
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt backprop  -dc fmnist -lr 0.01 -dt ${n_layers}
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc fmnist -lr 0.001 -s 6 -ld 1000 -dt ${n_layers}
run_plot -t hsic-solve -dc fmnist -e pdf
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt backprop  -dc cifar10 -lr 0.01 -dt ${n_layers}
run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc cifar10 -lr 0.001 -s 6 -ld 500 -dt ${n_layers}
run_plot -t hsic-solve -dc cifar10 -e pdf
mv assets/exp/fig4-hsic-solve-mnist-test-acc.pdf assets/exp/fig4-layer50-hsic-solve-mnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-fmnist-test-acc.pdf assets/exp/fig4-layer50-hsic-solve-fmnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-cifar10-test-acc.pdf assets/exp/fig4-layer50-hsic-solve-cifar10-test-acc.pdf
mv assets/exp/fig3-hsic-solve-actdist-mnist.pdf assets/exp/fig3-layer50-hsic-solve-actdist-mnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-fmnist.pdf assets/exp/fig3-layer50-hsic-solve-actdist-fmnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-cifar10.pdf assets/exp/fig3-layer50-hsic-solve-actdist-cifar10.pdf
