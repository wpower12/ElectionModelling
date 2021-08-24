import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

DATA_SET_DIR   = "data/prepared"
DATA_SET_LABEL = "reduced_features_02"

features_df = pd.read_csv("{}/{}/X_features.csv".format(DATA_SET_DIR, DATA_SET_LABEL), header=None)

FEATURE_LABELS = [
	# "B01001_001E",
	"ratio_age_u_5",
	"ratio_age_5_9",
	"ratio_age_10_14",
	"ratio_age_15_17",
	"ratio_age_18_19",
	"ratio_age_20",
	"ratio_age_21",
	"ratio_age_22_24",
	"ratio_age_25_29",
	"ratio_age_30_34",
	"ratio_age_35_39",
	"ratio_age_40_44",
	"ratio_age_45_49",
	"ratio_age_50_54",
	"ratio_age_55_59",
	"ratio_age_60_61",
	"ratio_age_62_64",
	"ratio_age_65_66",
	"ratio_age_67_69",
	"ratio_age_70_74",
	"ratio_age_75_79",
	"ratio_age_80_84",
	"ratio_age_85_u",
	"ratio_B01001A_001E",
	"ratio_B01001B_001E",
	"ratio_B01001C_001E",
	"ratio_B01001D_001E",
	"ratio_B01001E_001E",
	"ratio_B01001F_001E",
	"ratio_B01001G_001E",
	"ratio_B01001H_001E",
	"ratio_B01001I_001E"]
	# "norm_population",
	# "fips"]

features_df.columns = FEATURE_LABELS

plt.figure()
cor = features_df.corr()

sns.heatmap(cor, xticklabels=1, yticklabels=1, cmap=plt.cm.Reds)
plt.show()
