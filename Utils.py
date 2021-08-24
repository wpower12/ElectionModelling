import random
import torch
import math
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

import torch.nn.functional as f

RESULTS_FN_TEMP = "results/{}/{}"

def generateRandomSparseTensor(shape, density, max_value):
	num_items = math.floor(shape[0]*shape[1]*density)
	i, j, v = [], [], []
	for n in range(num_items):
		# Pick random u/v's in range
		i.append(random.randint(0, shape[0]-1))
		j.append(random.randint(0, shape[1]-1))
		v.append(random.randint(0, max_value))
	return torch.sparse_coo_tensor([i, j], v, shape, dtype=torch.float)


def writeMapToCSV(fn, src_map, headers):
	pathlib.Path(fn).parent.mkdir(exist_ok=True)

	# doing this manually bc i cant get pandas to do it right?
	# i mean its def me but lets just say its the panda.
	with open(fn, 'w') as f:
		f.write("{}\n".format(",".join(headers)))
		for key in src_map:
			val = src_map[key]
			f.write("{}, {}\n".format(key, val))


def writeTensorToCSV(fn, t):
	pathlib.Path(fn).parent.mkdir(exist_ok=True)
	
	with open(fn, 'w') as f:
		for row in t:
			row_str = ""
			for v in row:
				row_str += "{},".format(v)
			row_str = row_str[:-1]+"\n"
			f.write(row_str)


def writeListToCSV(fn, src_list):
	pathlib.Path(fn).parent.mkdir(exist_ok=True)
	save_df = pd.DataFrame(src_list)
	save_df.to_csv(fn, header=False, index=False)
	

class Logger:
	def __init__(self, file_name):
		pathlib.Path(file_name).parent.mkdir(exist_ok=True)
		self.fn = file_name

	def addValue(self, v):
		with open(self.fn, 'a') as f:
			f.write("{}\n".format(v))

	def addValues(self, vs):
		with open(self.fn, 'a') as f:
			for v in vs:
				f.write("{}\n".format(v))


class RMSLELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


def laplacianLoss(y_hat, y, L, alpha):
	term1   = f.normalize(y-y_hat, p=2.0, dim=0).sum()
	alpha_t = torch.transpose(alpha, 0, 1)
	term2   = torch.matmul(alpha_t, torch.matmul(L, alpha))
	return term1+term2


def generateTrainingFigure(x, y, data_label, file_name, fig_title):
	file_path = RESULTS_FN_TEMP.format(data_label, file_name)
	pathlib.Path(file_path).parent.mkdir(exist_ok=True)

	fig, ax = plt.subplots()
	ax.plot(x, y)
	ax.set(title=fig_title)

	fig.savefig(file_path)
	plt.close()


def adjFromCOO(coo_lists, N):
	adj = torch.zeros((N,N))
	E = len(coo_lists[0])
	for e in range(E):
		i = coo_lists[0][e]
		j = coo_lists[1][e]
		adj[i][j] = 1
	return adj


def lapFromAdj(adj):
	N = adj.shape[0]
	row_sums = torch.matmul(adj, torch.ones((N, 1)))
	D = torch.zeros((N,N))
	for i in range(N):
		D[i][i] = row_sums[i]
	L = D-adj
	return L