#!/bin/bash

run_hsicbt -cfg config/general-hsicbt.yaml   -tt hsictrain
run_hsicbt -cfg config/general-format.yaml   -tt format
run_hsicbt -cfg config/general-backprop.yaml -tt backprop
run_plot -t general -dc mnist -e pdf

