from torch_geometric.data import DataLoader

import torch
import pandas as pd

import Utils
import Data
import Models

TARGET_YEAR = 2020
RES_SAVE_LABEL = "reddit_test_00"

PER_TRAIN = 0.80

TARGET_SIZE = 1
NUM_HIDDEN  = 48
GCN_HIDDEN = 48

NUM_EPOCHS  = 200

# LEARNING_RATES = [0.0001, 0.00005]
LEARNING_RATES = [0.0005]
DECAYS         = [1e0]

TARGET_TYPE = 'per' # percent change. 
LABEL_STR_TEMP = "yearly_00_{}"
PER_TRAIN = 0.90

TARGET_YEARS = [2020, 2016, 2012]

graphs = []
for target_year in TARGET_YEARS:
	label = LABEL_STR_TEMP.format(target_year)
	# Data.generateProcessedDataset(target_year, label, PER_TRAIN, TARGET_TYPE)
	graph = Data.loadDataset(label)
	graphs.append(graph)


NUM_FEATURES = graphs[0].num_node_features

def run_MLP_pair(learing_rate, weight_decay):
	mlp_model     = Models.MLP(NUM_HIDDEN, NUM_FEATURES, TARGET_SIZE)
	mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learing_rate, weight_decay=weight_decay)
	mlp_criterion = torch.nn.MSELoss()

	def mlp_train_epoch():
		total_loss = 0
		dataloader = DataLoader(graphs)
		data_iter = iter(dataloader)
		for data in data_iter:
			mlp_model.train()
			mlp_optimizer.zero_grad()
			out  = mlp_model(data.x)
			loss = mlp_criterion(out[data.train_mask], data.y[data.train_mask])
			loss.backward()
			mlp_optimizer.step()
			total_loss += loss
			# print(loss)
		return total_loss/len(data_iter)

	def mlp_test_epoch():
		total_loss = 0
		dataloader = DataLoader(graphs)
		data_iter = iter(dataloader)
		for data in data_iter:
			mlp_model.eval()
			out  = mlp_model(data.x)
			loss = mlp_criterion(out[data.test_mask], data.y[data.test_mask])
			total_loss += losss
		return total_loss/len(data_iter)

	xs = []
	ys = []
	for epoch in range(0, NUM_EPOCHS):
		loss = mlp_train_epoch()
		xs.append(epoch)
		ys.append(loss)

	LR_STR = "{:.2e}".format(learing_rate).replace(".", "_").replace("+", "")
	WD_STR = "{:.2e}".format(weight_decay).replace(".", "_").replace("+", "")
	file_name = "mlp_lr_{}_wd_{}.png".format(LR_STR, WD_STR)
	label =  "MLP LR: {:.2e} WD: {:.2e}".format(learing_rate, weight_decay)
	Utils.generateTrainingFigure(xs, ys, RES_SAVE_LABEL, file_name, label)
	
	del mlp_model     
	del mlp_optimizer 
	del mlp_criterion 

def run_GNN_pair(learing_rate, weight_decay):
	ss_model     = Models.SimpleSkip(NUM_HIDDEN, NUM_FEATURES, GCN_HIDDEN, TARGET_SIZE)
	ss_optimizer = torch.optim.Adam(ss_model.parameters(), lr=learing_rate, weight_decay=weight_decay)
	ss_criterion = torch.nn.MSELoss()

	def ss_train_epoch():
		total_loss = 0
		dataloader = DataLoader(graphs)
		data_iter = iter(dataloader)
		for data in data_iter:
			ss_model.train()
			ss_optimizer.zero_grad()
			out  = ss_model(data.x, data.edge_index)
			loss = ss_criterion(out[data.train_mask], data.y[data.train_mask])
			loss.backward()
			ss_optimizer.step()
			total_loss += loss
		return total_loss/len(data_iter)

	def ss_test_epoch():
		total_loss = 0
		dataloader = DataLoader(graphs)
		data_iter = iter(dataloader)
		for data in data_iter:
			ss_model.eval()
			out  = ss_model(data.x, data.edge_index)
			loss = ss_criterion(out[data.test_mask], data.y[data.test_mask])
			total_loss += loss
		return total_loss/len(data_iter)

	xs = []
	ys = []
	for epoch in range(0, NUM_EPOCHS):
		loss = ss_train_epoch()
		xs.append(epoch)
		ys.append(loss)
		
	LR_STR = "{:.2e}".format(learing_rate).replace(".", "_").replace("+", "")
	WD_STR = "{:.2e}".format(weight_decay).replace(".", "_").replace("+", "")
	file_name = "gnn_lr_{}_wd_{}.png".format(LR_STR, WD_STR)
	label =  "GNN LR: {:.2e} WD: {:.2e}".format(learing_rate, weight_decay)
	Utils.generateTrainingFigure(xs, ys, RES_SAVE_LABEL, file_name, label)

	del ss_model     
	del ss_optimizer 
	del ss_criterion 

for lr in LEARNING_RATES:
	for wd in DECAYS:
		print("pair lr {}, wd {}".format(lr, wd))
		run_MLP_pair(lr, wd)
		run_GNN_pair(lr, wd)

