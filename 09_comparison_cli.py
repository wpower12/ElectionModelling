from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as f
import pandas as pd

import Utils
import Data
import Models

RES_SAVE_LABEL = "grid_all_3_01"
LABEL_STR_TEMP = "yearly_00_{}"

TARGET_SIZE = 1
NUM_HIDDEN  = 32
GCN_HIDDEN  = 48
SUB_REP_DIM = 2

TARGET_YEARS = [2020, 2016, 2012]

NUM_EPOCHS  = 300
LEARNING_RATE = 0.0002
WD = 0.01

graphs = []
for target_year in TARGET_YEARS:
	label = LABEL_STR_TEMP.format(target_year)
	graph = Data.loadDataset(label)
	graphs.append(graph)

NUM_FEATURES = graphs[0].num_node_features

reddit_activity = Data.loadRedditActivity(LABEL_STR_TEMP.format(2020))
reddit_activity = f.normalize(reddit_activity, dim=1)


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
	file_name = "MLP_lr_{}_wd_{}.png".format(LR_STR, WD_STR)
	fig_title =  "MLP LR: {:.2e} WD: {:.2e}".format(learing_rate, weight_decay)
	Utils.generateTrainingFigure(xs, ys, RES_SAVE_LABEL, file_name, fig_title)

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
	fig_title =  "GNN LR: {:.2e} WD: {:.2e}".format(learing_rate, weight_decay)
	Utils.generateTrainingFigure(xs, ys, RES_SAVE_LABEL, file_name, fig_title)

	del ss_model     
	del ss_optimizer 
	del ss_criterion 


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


print("pair lr {}, wd {}".format(LEARNING_RATE, WD))
run_MLP_pair(LEARNING_RATE, WD)
run_GNN_pair(LEARNING_RATE, WD)
run_RedGNN_pair(LEARNING_RATE, WD)
