import torch
import torch_geometric.utils as U
import torch.nn.parameter as P
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

### Baseline Temporal Skip Model
class MLP(torch.nn.Module):
	def __init__(self, num_hidden, dim_in, dim_out):
		super(MLP, self).__init__()
		self.H = num_hidden
		self.linear_1 = Linear(dim_in, self.H)
		self.linear_2 = Linear(self.H, dim_out)


	def forward(self, x):
		h = self.linear_1(x)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.tanh()

		h = self.linear_2(h)
		h = F.dropout(h, p=0.5, training=self.training)
		out = h.tanh()
		return out


### Simple GNN Model
class SimpleSkip(torch.nn.Module):
	def __init__(self, num_hidden, num_node_features, gcn_hidden, output_dim):
		super(SimpleSkip, self).__init__()

		# I ?think? these are values similar to what the paper uses.
		self.MLP_embed = MLP(num_hidden, num_node_features, gcn_hidden)
		self.GCN       = GCNConv(gcn_hidden, gcn_hidden)
		self.MLP_pred  = MLP(num_hidden, gcn_hidden, output_dim)
		

	def forward(self, x, edge_index):
		# Initial Embedding
		h = self.MLP_embed(x)
		# h = F.dropout(h, p=0.5, training=self.training)

		# First Hop
		h = self.GCN(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.tanh()

		# Prediction layer
		out = self.MLP_pred(h)
		return out


class AggregateSubreddits(torch.nn.Module):
	def __init__(self, activity, sub_rep_dim):
		super(AggregateSubreddits, self).__init__()

		num_subs = len(activity[0])

		self.S = activity
		self.R = P.Parameter(torch.rand((num_subs, sub_rep_dim)))


	def forward(self, x):
		# Aggregate the info fromthe subreddit reps
		# weighted by activity
		sub_agg = torch.matmul(self.S, self.R)

		# Concatenate that with x features to be the
		# initial input to the model. 
		h = torch.cat((x, sub_agg), 1)
		return h


class RedditGNN(torch.nn.Module):
	def __init__(self, reddit_activity, num_node_features, sub_rep_dim, num_hidden, gcn_hidden, output_dim):
		# reddit_activity, NUM_FEATURES, SUB_REP_DIM, NUM_HIDDEN, GCN_HIDDEN, TARGET_SIZE
		super(RedditGNN, self).__init__()

		self.AggSubs   = AggregateSubreddits(reddit_activity, sub_rep_dim)

		self.MLP_embed = MLP(num_hidden, num_node_features+sub_rep_dim, gcn_hidden)
		self.GCN1      = GCNConv(gcn_hidden, gcn_hidden)
		self.GCN2      = GCNConv(gcn_hidden, gcn_hidden)
		self.MLP_pred  = MLP(num_hidden, gcn_hidden, output_dim)


	def forward(self, x, edge_index):
		# Initial Embedding from this 'subreddit updated'
		# initial representation. The rest is the same as 
		# the other model. 
		h = self.AggSubs(x)

		h = self.MLP_embed(h) # We use the Embedding MLP as our 'update'
		h = F.dropout(h, p=0.5, training=self.training)

		# First Hop
		h = self.GCN1(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()
		
		# Second Hop
		h = self.GCN2(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()

		# Third Hop
		# h = self.GCN3(h, edge_index)
		# h = F.dropout(h, p=0.5, training=self.training)
		# h = h.relu()

		# Prediction layer
		out = self.MLP_pred(h)
		return out


class LinearCohesion(torch.nn.Module):
	def __init__(self, num_nodes, num_features):
		super(LinearCohesion, self).__init__()
		self.alpha = P.Parameter(torch.normal(0, 1, (num_nodes, 1)))
		self.beta  = P.Parameter(torch.normal(0, 1, (num_features, 1)))

	def forward(self, x):
		return torch.matmul(x, self.beta) + self.alpha