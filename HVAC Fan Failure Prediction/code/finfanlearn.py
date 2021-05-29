# %% [markdown]
# This file can be used for additional learning of the model

# %%
# Required imports and setting up
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import load_model
from datafunctions import convert_data, calc_rul, normalise, gen_labels, gen_sequence


np.random.seed(1010)
PYTHONHASHSEED = 0

model_path = "./finfanOUT/bin_model.h5"

if os.path.isfile(model_path):
    model = load_model(model_path)

# %%
# constant definitions
# these should be kept the same across all files
# the size of the blocks of data to be fed to the model, the larger the better
sequence_length = 50
# the number of days to predict failure within
w1 = pd.Timedelta(days=-30)
# scaler selection
minmax_scaler = preprocessing.MinMaxScaler()
# below should be all input variables given to the model
# list of all sensors used
sensor_cols = ["speed"]
# list of other variables
sequence_cols = ["status"]
sequence_cols.extend(sensor_cols)
# Data input: Read in the data and add to a list, for further processing
learn_df1 = pd.read_csv(
    "./finfanIN/Data_Extract_Train.txt", sep="\t", header=None, low_memory=False
)
learn_df2 = pd.read_csv(
    "./finfanIn/Data_Extract_Train2.txt", sep="\t", header=None, low_memory=False
)
learn_dfs = [learn_df1, learn_df2]

# %%
# converts the list of training dataframes to the correct format and adds it a single dataframe with an id column
# it cleans the data and merges all the time columns, calculates time until failure (assumed last point in the input data)
# and normalises the data using the scaler chosen, by default uses minmax which scales it to between 0 and 1
learn = None
for i, t in enumerate(learn_dfs):
    if learn is None:
        learn = normalise(calc_rul(convert_data(t, i), w1), minmax_scaler)
    else:
        learn = pd.concat(
            [learn, normalise(calc_rul(convert_data(t, i), w1), minmax_scaler)],
            copy=False,
        )
print(learn)
# %%
# generates the data chunks to be input into the model from the data previously formatted above
seq_gen = (
    list(gen_sequence(learn[learn["id"] == id], sequence_length, sequence_cols))
    for id in learn["id"].unique()
)

seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)

# %%
# generates the labels signifying if the failure will happen within the next 30 days
label_gen = [
    gen_labels(learn[learn["id"] == id], sequence_length, ["label1"])
    for id in learn["id"].unique()
]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape
# %%
# try to correctly predict outputs based on the given data
# save the model which best predicts
history = model.fit(
    seq_array,
    label_array,
    epochs=10,
    batch_size=200,
    validation_split=0.2,
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="min"
        ),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        ),
    ],
)
print(history.history.keys())

# %% [markdown]
# the below is to show the results of the training in a human readable format and is not required for the model to work
# %%
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["learn", "test"], loc="upper left")
plt.show()


# %%
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print("Accurracy: {}".format(scores[1]))


# %%
y_pred = (model.predict(seq_array, verbose=1, batch_size=200) > 0.5).astype("int32")
# y_pred = model.predict_classes(seq_array, verbose=1, batch_size=200)
y_true = label_array

test_set = pd.DataFrame(y_pred)

print("Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels")
cm = confusion_matrix(y_true, y_pred)
print(cm)


# %%
# compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print("precision = ", precision, "\n", "recall = ", recall)

# %%
