import pandas as pd
import numpy as np
from sklearn import preprocessing


def convert_data(df, index):
    """Given a dataframe that has 4 columns: time, speed, time, status
    merge the time columns and fill in missing values, removes all non number values


    Args:
        df (pandas dataframe): a pandas dataframe with [time, speed, time, status] columns
        index (any): a string or number to differentiate different input datas

    Returns:
        pandas dataframe: cleaned dataframe with collasped time columns
    """
    df[0] = pd.to_datetime(df[0], infer_datetime_format=True)
    df[2] = pd.to_datetime(df[2], infer_datetime_format=True)
    df.columns = ["time", "speed", "timea", "status"]
    df1 = df.drop(columns=["timea", "status"])
    df2 = df.drop(columns=["time", "speed"])
    df2.columns = ["time", "status"]
    df1 = df1.apply(pd.to_numeric, errors="coerce")
    df1 = df1.dropna()
    df2 = df2.apply(pd.to_numeric, errors="coerce")
    df2 = df2.dropna()
    df3 = df1.merge(df2, how="outer", on="time")
    df3["time"] = pd.to_datetime(df3["time"], infer_datetime_format=True)
    df3["id"] = index
    df3 = df3.pad()
    df3 = df3.bfill()
    print(df3)
    return df3


def calc_rul(df, window):
    """Calculates the time until failure for the training data assuming the last time value is failure


    Args:
        df (pandas dataframe): the dataframe to do the calculation on
        window (pandas timedelta): time before failure to use as window

    Returns:
        pandas dataframe: the original dataframe with new columns for within failure windows and remaining useful life
    """
    df["RUL"] = df["time"] - df.tail(1)["time"].values[0]
    df["label1"] = np.where(df["RUL"] >= window, 1, 0)
    return df


def normalise(df, scaler=None):
    """Normalises the dataframe given using the scaler given, defaults to MinMaxScaler is scaler is None

    Args:
        df (pandas dataframe): dataframe to normalise
        scaler (sklearn preproccessor): the scaler to use

    Returns:
        pandas dataframe: the normalise dataframe
    """
    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
    df1 = df.apply(pd.to_numeric)
    df1["time_norm"] = df1["time"]
    cols_normalise = df1.columns.difference(["time", "RUL", "label1", "id"])
    norm_df = pd.DataFrame(
        scaler.fit_transform(df1[cols_normalise]),
        columns=cols_normalise,
        index=df1.index,
    )
    join_df = df1[df1.columns.difference(cols_normalise)].join(norm_df)
    df2 = join_df.reindex(columns=df1.columns)
    print(df2)
    return df2


def gen_sequence(id_df, seq_length, seq_cols):
    """generates the sequence of chunks from the data

    Args:
        id_df (pandas dataframe): the dataframe to generate the sequence from
        seq_length (int): the size of each chunk
        seq_cols (list): the columns to include in the chunks

    Yields:
        generator: generator for the sequence
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(
        range(0, num_elements - seq_length), range(seq_length, num_elements)
    ):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    """generates labels for the sequence for if its within failure window

    Args:
        id_df (pandas dataframe): dataframe to generate labels for
        seq_length (int): the length of the sequences
        label (list): the column in the dataframe holding the labels

    Returns:
        list: list holding labels for the sequence
    """
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    return data_matrix[seq_length:num_elements, :]
