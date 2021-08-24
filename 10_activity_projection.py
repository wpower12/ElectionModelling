from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as f
import pandas as pd

import Utils
import Data
import Models

RES_SAVE_LABEL = "grid_all_3_01"
LABEL_STR_TEMP = "yearly_00_{}"
SAVE_PATH      = "data/prepared/yearly_00_2020/{}"

reddit_activity = Data.loadRedditActivity(LABEL_STR_TEMP.format(2020))
# reddit_activity = f.normalize(reddit_activity, dim=1)

proj = torch.matmul(reddit_activity.to_sparse(), reddit_activity.t())
# proj = proj.to_sparse()

torch.save(proj, SAVE_PATH.format('ra_proj'))

# scaling
diagonals = proj.diagonal() # Gets you a 1xN of the diags

print(diagonals)

scale_factors = torch.ones_like(proj)
for i in range(3114):
	for j in range(3114):
		scale_factors[i][j] = max(diagonals[i], diagonals[j])
		if scale_factors[i][j] == 0:
			scale_factors[i][j] = 1

scaled_proj = torch.div(proj, scale_factors)

torch.save(scaled_proj, SAVE_PATH.format('ra_proj_scaled'))

# remove diagonals.
for i in range(3114):
	scaled_proj[i][i] = 0

scaled_proj = scaled_proj.to_sparse()
nz_values   = scaled_proj.values()

print(nz_values.mean())
print(nz_values.std())

# Save the scaled project

