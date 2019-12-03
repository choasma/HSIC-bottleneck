#!/bin/bash

run_hsicbt -cfg config/varied-depth.yaml -dt 5  -ei 1
run_hsicbt -cfg config/varied-depth.yaml -dt 10 -ei 2
run_hsicbt -cfg config/varied-depth.yaml -dt 15 -ei 3
run_hsicbt -cfg config/varied-depth.yaml -dt 20 -ei 4
run_plot -t varied-depth -dc mnist -e pdf
