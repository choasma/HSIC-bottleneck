# bin

### run\_hsicbt
you could throw config file into the execution such as the following. this will create the log file under `assets/exp`
```sh
run_hsicbt -cfg config/whatever.yaml
```
however, if you want to overwrite the config, please refer to run_hsicbt get_args function. This will prevent multiple config files such as under grid-search experiments.
more examples are illustrated under `scripts` folder
```sh
# e.g., chaning batch-size and epochs
run_hsicbt -cfg config/whatever.yaml -ep 5 -bs 32
```

### run_plot
managing the plots according to each task
