import torch
import pandas as pd

import Utils
import Data
import Models

TARGET_YEAR = 2020
DATA_SRC_LABEL = "reduced_features_02"
RES_SAVE_LABEL = "grid_lr_wd_reduced_features_04"

PER_TRAIN = 0.80

TARGET_SIZE = 1
NUM_HIDDEN  = 48
GCN_HIDDEN = 48

NUM_EPOCHS  = 300

LEARNING_RATES = [0.001, 0.002, 0.005, 0.0001]
DECAYS         = [1e0, 1e-1]

# Data.generateProcessedDataset(TARGET_YEAR, LABEL, PER_TRAIN)
graph = Data.loadDataset(DATA_SRC_LABEL)
NUM_FEATURES = graph.num_node_features


def run_MLP_pair(learing_rate, weight_decay):
	mlp_model     = Models.MLP(NUM_HIDDEN, NUM_FEATURES, TARGET_SIZE)
	mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learing_rate, weight_decay=weight_decay)
	mlp_criterion = torch.nn.MSELoss()

	def mlp_train():
		mlp_model.train()
		mlp_optimizer.zero_grad()
		out  = mlp_model(graph.x)
		loss = mlp_criterion(out[graph.train_mask], graph.y[graph.train_mask])
		loss.backward()
		mlp_optimizer.step()
		return loss

	def mlp_test():
		mlp_model.eval()
		out  = mlp_model(graph.x)
		loss = mlp_criterion(out[graph.test_mask], graph.y[graph.test_mask])
		return loss

	xs = []
	ys = []
	for epoch in range(0, NUM_EPOCHS):
		loss = mlp_train()
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

	def ss_train():
		ss_model.train()
		ss_optimizer.zero_grad()
		out  = ss_model(graph.x, graph.edge_index)
		loss = ss_criterion(out[graph.train_mask], graph.y[graph.train_mask])
		loss.backward()
		ss_optimizer.step()
		return loss

	def ss_test():
		ss_model.eval()
		out  = ss_model(graph.x, graph.edge_index)
		loss = ss_criterion(out[graph.test_mask], graph.y[graph.test_mask])
		return loss

	xs = []
	ys = []
	for epoch in range(0, NUM_EPOCHS):
		loss = ss_train()
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