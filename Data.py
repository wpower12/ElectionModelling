import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

import Utils

ADJ_FN             = "data/raw/county_adjacency2010.csv"
STATIC_F_FN_TEMP   = "data/raw/{}_static_features_by_fips.csv"
STATIC_FEATURES_FN = "data/raw/static_features_by_fips.csv"
REGISTERED_ABS_FN  = "data/raw/2020_county_registered_abs.csv"
FULL_RETURNS_FN    = "data/raw/county_pres_2000-2020.csv"
SAVE_DIR_PATH      = "data/prepared/"
RACT_S_FN          = "data/raw/activity_daily_2020.csv"

SAVE_X_FN_TEMP     = "{}/{}/X_features.csv"
SAVE_Y_FN_TEMP     = "{}/{}/Y_targets.csv"
SAVE_COO_FN_TEMP   = "{}/{}/COO_list.csv"
SAVE_FIPS_FN_TEMP  = "{}/{}/fips_list.csv"
SAVE_TEST_FN_TEMP  = "{}/{}/test_mask.csv"
SAVE_TRAIN_FN_TEMP = "{}/{}/train_mask.csv"
SAVE_RDATA_FN_TEMP = "{}/{}/reddit_activity.csv"
SAVE_SUBMAP_TEMP   = "{}/{}/subreddit_list.csv"

LOAD_PROJ_TEMP     = "{}/{}/ra_proj.pt"
LOAD_PROJ_S_TEMP   = "{}/{}/ra_proj_scaled.pt"

def loadDataset(label, raw_tensors=False):
	with open(SAVE_X_FN_TEMP.format(SAVE_DIR_PATH, label), "r") as x_f:
		x_df = pd.read_csv(x_f, header=None)

	with open(SAVE_Y_FN_TEMP.format(SAVE_DIR_PATH, label), "r") as y_f:
		y_df = pd.read_csv(y_f, header=None)

	with open(SAVE_COO_FN_TEMP.format(SAVE_DIR_PATH, label), "r") as coo_f:
		coo_df = pd.read_csv(coo_f, header=None)

	with open(SAVE_TRAIN_FN_TEMP.format(SAVE_DIR_PATH, label), "r") as train_m_f:
		train_m_df = pd.read_csv(train_m_f, header=None)

	with open(SAVE_TEST_FN_TEMP.format(SAVE_DIR_PATH, label), "r") as test_m_f:
		test_m_df = pd.read_csv(test_m_f, header=None)	


	# COO Edge List Tensor. Reshaped. 
	coo_t   = torch.tensor(coo_df.values, dtype=torch.long)
	coo_t   = coo_t.reshape((2, len(coo_df.values)))

	# Features and Target Tensors.
	x_t     = torch.tensor(x_df.values, dtype=torch.float)
	y_t     = torch.tensor(y_df.values, dtype=torch.float)

	# Masks
	train_t = torch.tensor(train_m_df.values, dtype=torch.long)
	test_t  = torch.tensor(test_m_df.values,  dtype=torch.long)


	if raw_tensors:
		return x_t, y_t, coo_t, train_t, test_t
	else:
		return Data(x=x_t, y=y_t, edge_index=coo_t, train_mask=train_t, test_mask=test_t)


def loadRedditActivity(label):
	with open(SAVE_RDATA_FN_TEMP.format(SAVE_DIR_PATH, label), "r") as x_f:
		ract_df = pd.read_csv(x_f, header=None)
	return torch.tensor(ract_df.values, dtype=torch.float)


def loadProjectedActivity(label):
	p_t  = torch.load(LOAD_PROJ_TEMP.format(SAVE_DIR_PATH, label))
	ps_t = torch.load(LOAD_PROJ_S_TEMP.format(SAVE_DIR_PATH, label))
	return p_t, ps_t	


def generateProcessedDataset(target_year, label, per_train, target_type):
	# Creates the actual X, Y, COO lists
	# These are saved as text lists, along with maps
	# Then a separate method ingests those to build actual 
	# data objects for manipulation by ML tools. 
	fips_list = generateFIPSList()
	features  = generateYearlyFeatures(fips_list, target_year)
	targets   = generateElectionTargets(fips_list, target_year, target_type)

	# To do this, need an easy way to get an index from a FIPS.
	fips_map  = generateFIPSMap(fips_list)
	coo_list  = generateCOOList(fips_list, fips_map)

	# Next, we make some test train masks.
	train_mask, test_mask = generateTrainTestMasks(fips_list, per_train)

	# Finally, if this is a 2020 dataset, the reddit data. 
	if target_year == 2020:
		subreddit_map, reddit_data = getRedditActivity()

		Utils.writeTensorToCSV(SAVE_RDATA_FN_TEMP.format(SAVE_DIR_PATH, label), reddit_data)
		Utils.writeListToCSV(SAVE_SUBMAP_TEMP.format(SAVE_DIR_PATH, label), subreddit_map)

	Utils.writeListToCSV(SAVE_X_FN_TEMP.format(    SAVE_DIR_PATH, label), features)
	Utils.writeListToCSV(SAVE_Y_FN_TEMP.format(    SAVE_DIR_PATH, label), targets)
	Utils.writeListToCSV(SAVE_COO_FN_TEMP.format(  SAVE_DIR_PATH, label), coo_list)
	Utils.writeListToCSV(SAVE_FIPS_FN_TEMP.format( SAVE_DIR_PATH, label), fips_list)
	Utils.writeListToCSV(SAVE_TEST_FN_TEMP.format( SAVE_DIR_PATH, label), test_mask)
	Utils.writeListToCSV(SAVE_TRAIN_FN_TEMP.format(SAVE_DIR_PATH, label), train_mask)


def generateFIPSList():
	# Simply reads all of the data for the year and gets the 
	# SET of all available FIPS codes from the 'full' dataset
	# file. To make things easier to debug, I figure why not 
	# use a sorted list of the FIPS values? Enforces some 
	# kind of platform independant consistency. 
	full_df = pd.read_csv(FULL_RETURNS_FN, dtype={'county_fips': 'str'})
	full_df['county_fips'] = full_df['county_fips'].apply('{0:0>5}'.format)

	# Check this against the static features list?
	# Do we just leave them out? 
	static_df = pd.read_csv(STATIC_FEATURES_FN, dtype={'fips': 'str'})

	initial_fips_list = list(full_df['county_fips'].unique())
	fips_with_statics = []

	for fips in initial_fips_list:
		feature_row = static_df[static_df['fips'] == fips]
		if len(feature_row) == 1:
			fips_with_statics.append(fips)

	return sorted(fips_with_statics)


def generateFIPSMap(fips_list):
	fips_map = {}
	for i, fips in enumerate(fips_list):
		fips_map[fips] = i
	return fips_map


def generateYearlyFeatures(fips_list, target_year):
	static_df  = pd.read_csv(STATIC_F_FN_TEMP.format(target_year-1), dtype={'fips': 'str'})
	# reg_abs_df = pd.read_csv(REGISTERED_ABS_FN,  dtype={'FIPS': 'str'}) # 

	features = [] # Should be filled in the same order as the FIP list. 
	for fips in fips_list:
		c_count = 0 # Track 'touched' fips values, ones with features. 
		feature_row = static_df[static_df['fips']   == fips]
		# reg_abs_row = reg_abs_df[reg_abs_df['FIPS'] == fips]

		if len(feature_row) >= 1:
			c_count += 1
			# same as in prior method. 
			# NOTE - THIS IS WHERE YOU CHANGE HOW MUCH OF THE ACS TABLES YOU WANT. 
			static_features = list(feature_row.values[0][2:33])
			
			# full_features = static_features
			# print(reg_abs_count)
			# Here is where we would add things like 'registered voters' values. 	
			features.append(static_features)
		else:
			# Shouldn't happen anymore. 
			print('missing fips?', fips)

	return features


def generateElectionTargets(fips_list, target_year, target_type='abs'):
	full_df = pd.read_csv(FULL_RETURNS_FN, dtype={'county_fips': 'str'})
	full_df['county_fips'] = full_df['county_fips'].apply('{0:0>5}'.format)
	reg_abs_df = pd.read_csv(REGISTERED_ABS_FN,  dtype={'FIPS': 'str'})

	# What should the target be, exactly? Can we get a single number?
	# From weekly notes, we decided to use the draft concept of 
	# targeting the "change in 2 party vote spread".
	# (X_D-X_R-Y_D+Y_R)? 

	bad_rows = 0
	targets = []
	for fips in fips_list:
		# Need to get X_D, X_R, Y_D, Y_R

		county_df = full_df[full_df['county_fips'] == fips]
		year_X_df = county_df[county_df['year'] == target_year]
		year_Y_df = county_df[county_df['year'] == (target_year-4)]

		X_D = year_X_df[year_X_df['party'] == 'DEMOCRAT']['candidatevotes'].sum()
		X_R = year_X_df[year_X_df['party'] == 'REPUBLICAN']['candidatevotes'].sum()
		Y_D = year_Y_df[year_Y_df['party'] == 'DEMOCRAT']['candidatevotes'].sum()
		Y_R = year_Y_df[year_Y_df['party'] == 'REPUBLICAN']['candidatevotes'].sum()


		# Absolute Change in 'spread' between D and R, with + == more D, - == more R
		if target_type == 'abs':
			targets.append(X_D-X_R-Y_D+Y_R) # The absolute 'change in spread'
	
		# Percent of Registered Voters Change
		elif target_type == 'reg':
			reg_abs_row = reg_abs_df[reg_abs_df['FIPS'] == fips]
			if len(reg_abs_row) == 1:
				reg_abs_count = reg_abs_row.values[0][1]

				if reg_abs_count != 0.0:
					per_dX = (X_D-X_R)*100.0/reg_abs_count
					per_dY = (Y_D-Y_R)*100.0/reg_abs_count	
					targets.append(per_dX-per_dY) # redundant w.e.
				else:
					targets.append(0.0) #oof. 
					bad_rows += 1
			else:
				# Need to deal with this somehow. 
				# in the masks? ignore 0'd rows? idk. not many could 'naturally' be 0, right?
				targets.append(0.0)
				bad_rows += 1

		# Percent of Total Votes for Year Change. 
		elif target_type == 'per':
			total_X = X_D+X_R
			total_Y = Y_D+Y_R
			new_target = ((X_D-X_R)/total_X)-((Y_D-Y_R)/total_Y)

			if np.isnan(new_target):
				bad_rows += 1
				targets.append(0.0)
			else:
				targets.append(new_target)

	print("0'd rows: {}".format(bad_rows))
	return targets


def generateCOOList(fips_list, fips_map):
	ADJ_DTYPES = {'fipscounty': 'str', 'fipsneighbor': 'str'}
	adj_df = pd.read_csv(ADJ_FN, dtype=ADJ_DTYPES)

	coo_list   = []
	for fips in fips_list:
		u_idx = fips_map[fips]
		neighbors = adj_df[adj_df['fipscounty'] == fips]
		for neighbor in neighbors.values:

			v_fips = neighbor[3]
			if v_fips in fips_map:
				v_idx  = fips_map[v_fips]
				# for some reason we have to remove self links? idk. 
				if v_idx != u_idx:
					coo_list.append([u_idx, v_idx])
					coo_list.append([v_idx, u_idx])

	return coo_list


def generateTrainTestMasks(fips_list, percent_train):
	train_mask = np.random.rand(len(fips_list)) < percent_train
	test_mask  = ~train_mask
	return train_mask, test_mask


def getRedditActivity(norm_type='row'):
	fips_list = generateFIPSList()
	with open(RACT_S_FN, "r") as ract_f:
		ract_df = pd.read_csv(ract_f, dtype={'fips': 'str'})

	u_subs = list(ract_df['subreddit_id'].unique())

	fips_2_idx = {f: idx for idx, f in enumerate(fips_list)}
	subs_2_idx = {s: idx for idx, s in enumerate(u_subs)}

	num_counties = len(fips_list)
	num_subs     = len(u_subs)
	raw_tensor = torch.zeros((num_counties, num_subs))

	for row in ract_df.iterrows():
		raw  = row[1]
		fips = raw['fips'].zfill(5)
		sub  = raw['subreddit_id']
		val  = raw['activeusers']

		if fips in fips_list:
			c_idx = fips_2_idx[fips]
			s_idx = subs_2_idx[sub]
			raw_tensor[c_idx][s_idx] += val

	return u_subs, raw_tensor