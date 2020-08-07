""" An example showing how to plot from python scripts
"""

import matplotlib.pyplot as plt
import numpy as np

# let's say you produce two experiments
filepaths = [
    "./assets/logs/raw/200807_213841_hsic-solve-backprop-mnist.npy",
    "./assets/logs/raw/200807_214151_hsic-solve-backprop-mnist.npy",
]

# image legends, the length should be the same as filepaths
legends = [
    "a) Task 1",
    "b) Task 2",
]

# plotting each task
for i, f in enumerate(filepaths):
    a = np.load(f, allow_pickle=True)[()]['epoch_log_dict']['test_loss']
    plt.plot(a, label=legends[i])

# additional information
plt.xticks(np.arange(len(a)), np.arange(1, 6))
plt.ylabel("\"test\" accuracy")
plt.xlabel("training epoch")
plt.title("Just an example")
plt.legend(fontsize=8)
plt.savefig("example.png")
print("Saved", "./example.png")
