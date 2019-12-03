#!/bin/bash

run_hsicbt -cfg config/sigma-combined.yaml -s 1  -ei 1 -tt hsictrain
run_hsicbt -cfg config/sigma-combined.yaml -s 5  -ei 2 -tt hsictrain
run_hsicbt -cfg config/sigma-combined.yaml -s 10 -ei 3 -tt hsictrain
run_hsicbt -cfg config/sigma-combined.yaml -ei 1 -tt format -lr 0.004
run_hsicbt -cfg config/sigma-combined.yaml -ei 2 -tt format -lr 0.004
run_hsicbt -cfg config/sigma-combined.yaml -ei 3 -tt format -lr 0.004
run_hsicbt -cfg config/sigma-combined.yaml -ei 1 2 3 -tt format -lr 0.004
run_plot -t sigma-combined -dc mnist -e pdf
