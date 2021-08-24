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

RUNS_PER_MODEL = 10
ITERATIONS = 1000

x_t, y_t, coo_t, train_t, test_t = Data.loadDataset(LABEL_STR, raw_tensors=True)
proj_t, proj_scaled_t            = Data.loadProjectedActivity(LABEL_STR)


def runModelBatch(L, R):
	total_loss = 0
	for r in range(R):
		lc_Model = Models.LinearCohesion(N, NUM_FEATURES)
		lc_Optim = torch.optim.Adam(lc_Model.parameters(), lr=0.1)
		for _ in range(ITERATIONS):
			lc_Model.train()
			lc_Optim.zero_grad()
			output = lc_Model(x_t)
			loss = Utils.laplacianLoss(output, y_t, L, lc_Model.alpha.data)
			loss.backward()
			lc_Optim.step()
		total_loss += loss
	return total_loss / R 


# geographic edges
geo_adj = Utils.adjFromCOO(coo_t, N)
geo_L   = Utils.lapFromAdj(geo_adj)

print("geo  loss ave: {}".format(runModelBatch(geo_L, RUNS_PER_MODEL)))

# reddit edges
thresholds = [0, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5]
ra_adjs = [torch.zeros((N,N)) for n in range(len(thresholds))]
 
for i in range(N):
	for j in range(N):
		for threshold_id, threshold in enumerate(thresholds):
			if proj_scaled_t[i][j] > threshold:
				ra_adjs[threshold_id][i][j] = 1

ra_Ls = [Utils.lapFromAdj(ra_adj) for ra_adj in ra_adjs]

for ra_idx, ra_L in enumerate(ra_Ls):
	ra_loss = runModelBatch(ra_L, RUNS_PER_MODEL)
	ra_num_edges = torch.sum(ra_adjs[ra_idx])
	print("thresh: {} num_edges: {} loss: {}".format(thresholds[ra_idx], ra_num_edges, ra_loss))