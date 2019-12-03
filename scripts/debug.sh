#!/bin/bash

run_hsicbt -cfg config/general-backprop.yaml -ep 1
run_hsicbt -cfg config/general-hsicbt.yaml -ep 1
run_hsicbt -cfg config/general-format.yaml -ep 1
run_hsicbt -cfg config/hsicsolve.yaml -ep 1
run_hsicbt -cfg config/needle.yaml -ep 1
run_hsicbt -cfg config/resconv-backprop.yaml -ep 1
run_hsicbt -cfg config/resconv-format.yaml -ep 1
run_hsicbt -cfg config/resconv-hsicbt.yaml -ep 1
run_hsicbt -cfg config/sigma-combined.yaml -ep 1
run_hsicbt -cfg config/varied-activation.yaml -ep 1
run_hsicbt -cfg config/varied-depth.yaml -ep 1
run_hsicbt -cfg config/varied-dim.yaml -ep 1
run_hsicbt -cfg config/varied-epoch.yaml -ep 1
