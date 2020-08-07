""" An example how to plot animation like in README
Please replace the regex string for your batch log output
"""

from hsicbt.utils import plot
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_needle_distribution():
    regex = "./assets/activation/raw/070820_152112_needle-hsictrain-mnist-*.npy"
    filepaths = sorted(glob.glob(regex))
    for idx, filepath in enumerate(filepaths):
        plot.plot_1d_activation_kde(filepath)
        plt.title("Epoch {}".format(idx), fontsize=30)
        plot.save_figure(filepath[:-3]+"png")

def plot_batch_hsicsolve():
    regex = "./assets/activation/raw/200807_180226_hsic-solve-hsictrain-mnist_batch-*.npy"
    filepaths = sorted(glob.glob(regex))[::2]
    for idx, filepath in enumerate(filepaths):
        title = "Iteration {} @Epoch 1".format(idx*2)
        plot.plot_activation_distribution(filepath, title)
        out_path = filepath[:-8]+"{:04d}.png".format(idx)    
        plot.save_figure(out_path)
    
if __name__ == "__main__":
    plot_needle_distribution()
    plot_batch_hsicsolve()
    # Then use imagemagic command to make gif animation
    # convert -delay 2 /path/to/name.*.png /path/out.gif
