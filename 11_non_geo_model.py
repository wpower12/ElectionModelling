import torch
import torch.nn.parameter as P
import torch.nn.functional as f
import pandas as pd

import Utils
import Data
import Models

RES_SAVE_LABEL = "grid_all_3_01"
LABEL_STR      = "yearly_00_2020"
N = 3114
NUM_FEATURES = 31  # NUM_FEATURES 
ITERATIONS = 1000

x_t, y_t, coo_t, train_t, test_t = Data.loadDataset(LABEL_STR, raw_tensors=True)
proj_t, proj_scaled_t            = Data.loadProjectedActivity(LABEL_STR)

# Need to get the coo_t into an adjacecny form.
adj = torch.zeros((N,N))
E = len(coo_t[0])
for e in range(E):
	i = coo_t[0][e]
	j = coo_t[1][e]
	adj[i][j] = 1

# get the graph laplacian. 
row_sums = torch.matmul(adj, torch.ones((N, 1)))
D = torch.zeros((N,N))
for i in range(N):
	D[i][i] = row_sums[i]
L = D-adj


class LinearCohesion(torch.nn.Module):
	def __init__(self):
		super(LinearCohesion, self).__init__()
		self.alpha = P.Parameter(torch.normal(0, 1, (N, 1)))
		self.beta  = P.Parameter(torch.normal(0, 1, (NUM_FEATURES, 1)))

	def forward(self, x):
		return torch.matmul(x, self.beta) + self.alpha


def laplacianLoss(y_hat, y, L, alpha):
	term1   = f.normalize(y-y_hat, p=2.0, dim=0).sum()
	alpha_t = torch.transpose(alpha, 0, 1)
	term2   = torch.matmul(alpha_t, torch.matmul(L, alpha))
	return term1+term2

lc_Model = LinearCohesion()
lc_Optim = torch.optim.Adam(lc_Model.parameters(), lr=0.1)

for _ in range(ITERATIONS):
	lc_Model.train()
	lc_Optim.zero_grad()
	output = lc_Model(x_t)
	loss = laplacianLoss(output, y_t, L, lc_Model.alpha.data)
	loss.backward()
	lc_Optim.step()

print(loss)







