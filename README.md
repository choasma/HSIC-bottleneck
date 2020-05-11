

# HSIC-Bottleneck
This is the released repo for our work entitled `The HSIC Bottleneck: Deep Learning without Back-Propagation`. All the experiments were produced by this repository.

# Updates
Our work has been accepted by AAAI2020 international conference.

### Environment
- pytorch-1.1.0
- torchvision-0.3.0
- numpy-1.16.4
- scipy-1.3.0
- tqdm-4.33.0
- yaml-5.1.1

### Usage

Each run is according to the specific task generating figure in the paper. To reproduce all the experiments that we have in the paper, you could run our batch script by the following instruction:
```sh
# in bash
git clone git@gitlab.com:gladiator8072/hsic-bottleneck.git 
source env.sh
batch.sh
```

After the batch training, you'll get the following result in your `assets/exp` folder, and please compare to our sample results under `assets/sample` folder:
For more information, please visit the running procedure page (config/READMD.md [link](config/README.md)) and (bin/README.md [link](bin/README.md)) for more information.
```sh
# there's fig* context at beginning for convenience 
[mawand@machine HSIC-bottleneck]$ ls -l assets/exp | awk '{print $9}'
fig2a-varied-activation-hsic_xz-mnist.pdf
fig2b-varied-activation-hsic_yz-mnist.pdf
fig2c-varied-activation-acc-mnist.pdf
fig2d-varied-depth-hsic_xz-mnist.pdf
fig2e-varied-depth-hsic_yz-mnist.pdf
fig2f-varied-depth-acc-mnist.pdf
fig3a-needle-1d-dist-backprop.pdf
fig3b-needle-1d-dist-hsictrain.pdf
fig4-hsic-solve-actdist-cifar10.pdf
fig4-hsic-solve-actdist-fmnist.pdf
fig4-hsic-solve-actdist-mnist.pdf
fig5-hsic-solve-cifar10-train-acc.pdf
fig5-hsic-solve-fmnist-train-acc.pdf
fig5-hsic-solve-mnist-train-acc.pdf
fig6a-varied-epoch-acc-mnist.pdf
fig6b-varied-epoch-loss-mnist.pdf
fig7a-varied-dim-acc-mnist.pdf
fig7b-sigma-combined-mnist-sigmacomb-train-acc.pdf
```

### Note
This repo is still under developing of documentation, but let me know if you need further information by emailing me or create issues

### Cite
Please cite our work if it is relevant to your research work, thanks!

```
@article{Ma2019TheHB,
  title={The HSIC Bottleneck: Deep Learning without Back-Propagation},
  author={Wan-Duo Ma and J. P. Lewis and W. Bastiaan Kleijn},
  journal={ArXiv},
  year={2019},
  volume={abs/1908.01580}
}
```
