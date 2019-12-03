#!/bin/bash

run_hsicbt -cfg config/varied-dim.yaml -d 8  -ei 1 -tt hsictrain
run_hsicbt -cfg config/varied-dim.yaml -d 32 -ei 2 -tt hsictrain
run_hsicbt -cfg config/varied-dim.yaml -d 64 -ei 3 -tt hsictrain
run_hsicbt -cfg config/varied-dim.yaml -d 8  -ei 1 -ep 5 -tt format -lr 0.01
run_hsicbt -cfg config/varied-dim.yaml -d 32 -ei 2 -ep 5 -tt format -lr 0.01
run_hsicbt -cfg config/varied-dim.yaml -d 64 -ei 3 -ep 5 -tt format -lr 0.01
run_plot -t varied-dim -dc mnist -e pdf
