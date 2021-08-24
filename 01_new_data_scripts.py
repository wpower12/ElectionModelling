import pandas as pd
import Data

TARGET_YEAR = 2020
TARGET_TYPE = 'per' # percent change. 
LABEL = "reduced_features_02"
PER_TRAIN = 0.90

Data.generateProcessedDataset(TARGET_YEAR, LABEL, PER_TRAIN, TARGET_TYPE)
graph = Data.loadDataset(LABEL)
print(graph)

