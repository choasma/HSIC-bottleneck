# Task Script

Each config file represents the single task, which might have several trainings according to the paper figures. You could simply run `batch.sh` to reproduce all the figures from our paper, or either go to each task script described in batch.sh for further research.


# General Procedure

#### pre-action
- go to the project root directory
- setting the environment `source env.sh`
- run command `task_`

#### runtime
- load config `[config_path]`
- training, and save the logs under `./assets/logs` (optional)
- save figures in `./assets/exp`

# Tasks

#### varied-activation (fig2a-c)
- note
nHSIC monitoring in backprop training. It shows nHSIC(X,Z\_L), nHSIC(Y,Z\_L) would decrease and increase
respectively. As the evidence the network learn the dependency from output and reduce it from input
- commands
```sh
task_varied-act.sh
```

#### varied-depth (fig2d-f)
- note
nHSIC monitoring in backprop training. It shows nHSIC(X,Z\_L), nHSIC(Y,Z\_L) would decrease and increase
respectively. As the evidence the network learn the dependency from output and reduce it from input
- commands
```sh
task_varied-dim.sh
```

#### needle [DEPRECATED]
- note
Originally placed in our first arxiv in fig3. This experiment shows how well the HSIC-trained network
separates the classed signals in the scalared network output (tanh case)
- commands
```sh
task_needle.sh
```

#### hsicsolve (fig3, fig4)
- note
Unformat-training, where we use HSIC-trained network to solve the classification problem. The output
of the network is no longer ordered vector as in one-hot label matrix. The particular image category
might go to some other entry.
- commands
```sh
task_hsicsolve.sh
```

#### varied-epoch (fig5a-b)
- note
The boosting test which the long HSIC-trained network allows faster convergence of format-train.
- commands
```sh
task_varied-ep.sh
```

#### varied-dim (fig6a)
- note
The capacity of HSIC-trained network experiments. The aim of this task is we hope large network can hold 
more information from the input, making the format-training better.
- commands
```sh
task_varied-dim.sh
```

#### sigma-combined (fig6b)
- note
The capacity of HSIC-trained network experiments. First of all process 3 HSIC-trained network with different 
sigma scale and produce 3 format-training results. Then load and average those 3 networks for format-training, 
should be better than those individually
- commands
```sh
task_combsig.sh
```

#### conv-based resnet (fig7)
- note
The HSIC-bottlneck on the ResNet based architecture, where the objective are applied on each residual block output
- commands
```sh
task_resconv.sh
```
