#!/bin/bash

"""
the example running same experiment with different randomseed
turn on the code block if needed
"""

for s in {1..10}
do
    echo "Task:${s}, Seed:${RANDOM}"

    # # # needle-case
    # run_hsicbt -cfg config/needle.yaml -tt hsictrain -ep 5 -sd ${RANDOM}
    # run_plot -t needle -dc mnist -e pdf -tt hsictrain -ft 'random seed:'"$RANDOM"

    # # # unformat-training
    # run_hsicbt -cfg config/hsicsolve.yaml -tt hsictrain -dc mnist -sd ${RANDOM}
    # run_plot -t hsic-solve -dc mnist -e pdf -ft 'random seed:'"$RANDOM"
    
done
