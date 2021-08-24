import torch
import pandas as pd

import Utils
import Data
import Models

TARGET_YEAR = 2020
LABEL = "reg_per_00"
PER_TRAIN = 0.80

TARGET_SIZE = 1
NUM_HIDDEN  = 32
GCN_HIDDEN = 16

NUM_EPOCHS  = 200

LEARNING_RATE = 0.00001
WEIGHT_DECAY  = 1e-5

# Data.generateProcessedDataset(TARGET_YEAR, LABEL, PER_TRAIN)
graph = Data.loadDataset(LABEL)
NUM_FEATURES = graph.num_node_features


## MLP - No Edges.
mlp_model     = Models.MLP(NUM_HIDDEN, NUM_FEATURES, TARGET_SIZE)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
mlp_criterion = torch.nn.MSELoss()

print(mlp_model)

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

print("mlp - initial loss - {}".format(mlp_test()))
xs = []
ys = []
for epoch in range(0, NUM_EPOCHS):
	loss = mlp_train()
	xs.append(epoch)
	ys.append(loss)
	# print("mlp - epoch {:0>} - {}".format(epoch, loss))
	# Figure out saving to a figure or table.
 
Utils.generateTrainingFigure(xs, ys, LABEL, "mlp.png", "Basic MLP")
print("mlp - final test loss: {}".format(mlp_test()))

## GNN - With edges!
ss_model     = Models.SimpleSkip(NUM_HIDDEN, NUM_FEATURES, GCN_HIDDEN, TARGET_SIZE)
ss_optimizer = torch.optim.Adam(ss_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ss_criterion = torch.nn.MSELoss()

print(ss_model)

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

print("ss - initial loss - {}".format(ss_test()))

xs = []
ys = []
for epoch in range(0, NUM_EPOCHS):
	loss = ss_train()
	xs.append(epoch)
	ys.append(loss)
	# print("ss - epoch {:0>} - {}".format(epoch, loss))
	# Figure out saving to a figure or table.

Utils.generateTrainingFigure(xs, ys, LABEL, "gnn.png", "Basic GNN")
print("ss - final test loss: {}".format(ss_test()))