"""
Question on block coordinate descent #10
https://github.com/choasma/HSIC-bottleneck/issues/10
(1) why this part does not use backprop? (the following part involves loss.backward() and optimizer.step())
(2) what the "block" here means in block coordinate descent? Is the neurons in the same layer considered as in the same group?
"""
import sys
sys.path.append("source")
import torch
import numpy as np
from hsicbt.utils import misc
from hsicbt.model.mhlinear import ModelLinear

# # # our model
model = ModelLinear(last_hidden_width=10)
print(model)

# # # Preparation
batch_size = 32
train_x = torch.randn(batch_size, 784)
train_y = torch.randint(0,10,(batch_size,)).long()
criterion = torch.nn.CrossEntropyLoss()
output, hiddens = model(train_x)
idx_range = []



print("========== Proposed approach ============")
layer_idx = 3 # let's say third layer
it = 0 # It's ugly, the aim is trying to query the parameters of the model at each layer, which is skip 2 because weight and bias
for i in range(len(hiddens)):
    idx_range.append(np.arange(it, it+2).tolist())
    it += 2
params, param_names = misc.get_layer_parameters(model=model, idx_range=idx_range[layer_idx])
optimizer = torch.optim.SGD(params, lr=0.1, momentum=.9, weight_decay=0.001) # we only expose the weights at layer_idx to optimizer
loss = criterion(output, train_y)
loss.backward()

# # # Check before&after weight update
norm_before_step = []
for p in model.parameters():
    norm_before_step.append(torch.norm(p).item())
optimizer.step() # let's apply weights on model
norm_after_step = []
for p in model.parameters():
    norm_after_step.append(torch.norm(p).item())
# # # Difference checking
print(f"Diff of the model weight and bias (Only layer:{layer_idx} are updated)")
print([val[0]-val[1] for val in zip(norm_before_step, norm_after_step) ])


print("========== Standard backprop ============")
model = ModelLinear()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=.9, weight_decay=0.001)
output, hiddens = model(train_x)
loss = criterion(output, train_y)
loss.backward()

norm_before_step = []
for p in model.parameters():
    norm_before_step.append(torch.norm(p).item())
optimizer.step()
norm_after_step = []
for p in model.parameters():
    norm_after_step.append(torch.norm(p).item())

print("Diff of the model weight and bias in backprop (All weights should be changed)")
print([val[0]-val[1] for val in zip(norm_before_step, norm_after_step) ])
