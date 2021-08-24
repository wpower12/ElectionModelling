from torch_geometric.data import DataLoader


import torch
import torch.nn.functional as f
import pandas as pd

import Utils
import Data
import Models

TARGET_YEAR = 2020
RES_SAVE_LABEL = "grid_3_cycles_03"

PER_TRAIN = 0.80

TARGET_SIZE = 1
NUM_HIDDEN  = 48
GCN_HIDDEN = 48

SUB_REP_DIM = 2

NUM_EPOCHS  = 200

# LEARNING_RATES = [0.0001, 0.00005]
LEARNING_RATES = [0.0005]
DECAYS         = [1e0]

LABEL_STR_TEMP = "yearly_00_{}"
DATA_LABEL = "yearly_00_2020"
TARGET_YEARS = [2020, 2016]

graphs = []
for target_year in TARGET_YEARS:
	label = LABEL_STR_TEMP.format(target_year)
	# Data.generateProcessedDataset(target_year, label, PER_TRAIN, TARGET_TYPE)
	graph = Data.loadDataset(label)
	graphs.append(graph)

NUM_FEATURES = graphs[0].num_node_features

sub_map, reddit_activity = Data.getRedditActivity()
reddit_activity = f.normalize(reddit_activity, dim=1)

def run_RedGNN_pair(learing_rate, weight_decay):
	# reddit_activity, num_node_features, sub_rep_dim, output_dim
	rgnn_model     = Models.RedditGNN(reddit_activity, NUM_FEATURES, SUB_REP_DIM, NUM_HIDDEN, GCN_HIDDEN, TARGET_SIZE)
	rgnn_optimizer = torch.optim.Adam(rgnn_model.parameters(), lr=learing_rate, weight_decay=weight_decay)
	rgnn_criterion = torch.nn.MSELoss()

	def rgnn_train_epoch():
		total_loss = 0
		dataloader = DataLoader(graphs)
		data_iter = iter(dataloader)
		for data in data_iter:
			rgnn_model.train()
			rgnn_optimizer.zero_grad()
			out  = rgnn_model(data.x, data.edge_index)
			loss = rgnn_criterion(out[data.train_mask], data.y[data.train_mask])
			loss.backward()
			rgnn_optimizer.step()
			total_loss += loss
		return total_loss/len(data_iter)

	def rgnn_test_epoch():
		total_loss = 0
		dataloader = DataLoader(graphs)
		data_iter = iter(dataloader)
		for data in data_iter:
			rgnn_model.eval()
			out  = rgnn_model(data.x, data.edge_index)
			loss = rgnn_criterion(out[data.test_mask], data.y[data.test_mask])
			total_loss += loss
		return total_loss/len(data_iter)


	xs = []
	ys = []
	for epoch in range(0, NUM_EPOCHS):
		loss = rgnn_train_epoch()
		print(loss)
		xs.append(epoch)
		ys.append(loss)
		
	LR_STR = "{:.2e}".format(learing_rate).replace(".", "_").replace("+", "")
	WD_STR = "{:.2e}".format(weight_decay).replace(".", "_").replace("+", "")
	file_name = "rgnn_lr_{}_wd_{}.png".format(LR_STR, WD_STR)
	fig_title =  "RedGNN LR: {:.2e} WD: {:.2e}".format(learing_rate, weight_decay)
	Utils.generateTrainingFigure(xs, ys, RES_SAVE_LABEL, file_name, fig_title)

	del rgnn_model     
	del rgnn_optimizer 
	del rgnn_criterion 

for lr in LEARNING_RATES:
	for wd in DECAYS:
		print("pair lr {}, wd {}".format(lr, wd))
		run_RedGNN_pair(lr, wd)


