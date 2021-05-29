# %% [markdown]
# this file is to generate predictions based on input data

# %%
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from keras.models import load_model
from datafunctions import convert_data, normalise, gen_sequence

np.random.seed(1010)
PYTHONHASHSEED = 0

model_path = "./finfanOUT/bin_model.h5"

# load the model
if os.path.isfile(model_path):
    estimator = load_model(model_path)

# %% [markdown]
# Set up the data to generate predictions on

# %%
# constant definitions take these from what the initial model was trained with
sequence_length = 50
minmax_scaler = preprocessing.MinMaxScaler()
# below should be all input variables given to the model
# list of all sensors used
sensor_cols = ["speed"]
# list of other variables
sequence_cols = ["status"]
sequence_cols.extend(sensor_cols)
# read in data to predict on
test_df2 = pd.read_csv(
    "./finfanIn/Data_Extract_Train2.txt", sep="\t", header=None, low_memory=False
)
test_dfs = [test_df2]
# %%
# converts the list of training dataframes to the correct format and adds it a single dataframe with an id column
# it cleans the data and merges all the time columns
# and normalises the data using the scaler chosen, by default uses minmax which scales it to between 0 and 1
test_df = None
for i, t in enumerate(test_dfs):
    if test_df is None:
        test_df = normalise(convert_data(t, i), minmax_scaler)
    else:
        test_df = pd.concat(
            [test_df, normalise(convert_data(t, i), minmax_scaler)],
            copy=False,
        )
print(test_df)

# %%
# generates the data chunks to be input into the model from the data previously formatted above
seq_gen = (
    list(gen_sequence(test_df[test_df["id"] == id], sequence_length, sequence_cols))
    for id in test_df["id"].unique()
)

seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
# %%
# generate predictions for the input data comes out as a number between 0 and 1 for the confidence that it is within window until failure
predictions = estimator.predict(seq_array, verbose=1, batch_size=200)
print(predictions)

# %%
