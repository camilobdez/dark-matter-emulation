# %%
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

import veccs.orderings
from batram.legmods import Data, SimpleTM

# %% [Load Data]
# Load locations and observations from CSV files.
locs = pd.read_csv("locs.csv", header=None).values  # Expect shape (4096, 2)
obs = torch.as_tensor(pd.read_csv("stacked.csv", header=None).to_numpy(), dtype=torch.float32)
obs = obs.T  # Now obs is (200, 4096): 200 images, each 4096 pixels

ord = np.lexsort((locs[:, 1], locs[:, 0]))
locs = locs[ord]
obs = obs[:, ord]


print(f"Locations array dimension: {locs.shape}")
print(f"Observations array dimension: {obs.shape}")

# %% [Visualize a few images]
gspec = {"wspace": 0.1, "hspace": 0.1}
fig, ax = plt.subplots(40, 5, figsize=(15, 6*20), gridspec_kw=gspec, squeeze=True)
vmin, vmax = obs.min(), obs.max()
for i in range(200):
    _ = ax[i // 5, i % 5]
    im = _.imshow(obs[i].reshape(64, 64), vmin=vmin, vmax=vmax)
    _.set_xticks([])
    _.set_yticks([])
fig.subplots_adjust(right=0.9)
cbar = fig.add_axes([0.125, 0.05, 0.775, 0.045])
fig.colorbar(im, cax=cbar, orientation="horizontal")
plt.show()


# %%
obs_mean = obs.mean(dim=0, keepdim=True)
obs_std = obs.std(dim=0, keepdim=True)
obs = (obs - obs_mean) / obs_std

train_obs = obs[:160, :]
test_obs  = obs[160:200, :]

order = veccs.orderings.maxmin_cpp(locs)
locs_ordered = locs[order, :]

train_obs_ordered = train_obs[:, order]
test_obs_ordered  = test_obs[:, order]

largest_conditioning_set = 100
nn = veccs.orderings.find_nns_l2(locs_ordered, largest_conditioning_set)

locs_tensor = torch.as_tensor(locs_ordered, dtype=torch.float32)

train_data = Data.new(locs_tensor, train_obs_ordered, torch.as_tensor(nn))
test_data  = Data.new(locs_tensor, test_obs_ordered, torch.as_tensor(nn))

# %%
tm = SimpleTM(train_data, theta_init=None, linear=False, smooth=1.5, nug_mult=4)

# %%
params = list(tm.nugget.parameters())
print(params)


# %%
nsteps = 100
opt = torch.optim.Adam(tm.parameters(), lr=0.0001, weight_decay=4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps, eta_min=0.0001)
res = tm.fit(nsteps, 0.0001, test_data=test_data, optimizer=opt, scheduler=sched, batch_size=64)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.7})
res.plot_loss(axs[0], use_inset=False)
res.plot_loss(axs[1], use_inset=True)
plt.show()

# %%
res.plot_params()
plt.show()

# %%
res.plot_neighbors()
plt.show()

# %%
# Generate an unconditional new sample.
new_sample = tm.cond_sample()[0, ...]
# Reorder new_sample back to the original spatial order.
re_ordered_sample = torch.zeros_like(new_sample)
re_ordered_sample[..., order] = new_sample

# To convert back to the original scale, you might "undo" the normalization:
simulated = re_ordered_sample * obs_std + obs_mean

plt.imshow(simulated.reshape(64, 64), vmin=vmin, vmax=vmax, origin="upper")
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.title("New Sample")
plt.show()


# %%
i = 23  # e.g., the first test image in the test_data
test_field_ordered = test_data.response[i].clone()  # shape (4096,)

# PARTIAL OBSERVATION: fix the first 100 pixels
# Make sure these pixels are in the model's ordering
x_fix = test_field_ordered[:1000]

# Generate 2 conditional samples from the model
cond_samples = tm.cond_sample(x_fix=x_fix, num_samples=2)

# Reorder all fields back to the original (pre-maximin) order
# 1) The full test image
test_field_reordered = torch.zeros_like(test_field_ordered)
test_field_reordered[..., order] = test_field_ordered

# 2) The conditional samples
cond1 = cond_samples[0]
cond2 = cond_samples[1]

cond1_reordered = torch.zeros_like(cond1)
cond2_reordered = torch.zeros_like(cond2)
cond1_reordered[..., order] = cond1
cond2_reordered[..., order] = cond2


test_field_un = test_field_reordered * obs_std + obs_mean
cond1_un = cond1_reordered * obs_std + obs_mean
cond2_un = cond2_reordered * obs_std + obs_mean



# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original test image
im0 = axes[0].imshow(test_field_un.reshape(64, 64), vmin=vmin, vmax=vmax, origin="upper")
axes[0].set_title("Test data")
axes[0].set_xticks([])
axes[0].set_yticks([])

# Conditional sample 1
im1 = axes[1].imshow(cond1_un.reshape(64, 64), vmin=vmin, vmax=vmax, origin="upper")
axes[1].set_title("Conditional sample 1")
axes[1].set_xticks([])
axes[1].set_yticks([])

# Conditional sample 2
im2 = axes[2].imshow(cond2_un.reshape(64, 64), vmin=vmin, vmax=vmax, origin="upper")
axes[2].set_title("Conditional sample 2")
axes[2].set_xticks([])
axes[2].set_yticks([])

# Single colorbar for all  subplots
# left, bottom, width, height in figure coordinates
cbar_ax = fig.add_axes([0.02, -0.1, 0.95, 0.1])
fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")
plt.tight_layout()
plt.show()


# %%



