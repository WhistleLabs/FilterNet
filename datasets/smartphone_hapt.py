# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

""" Loads the smartphone human activity and posture detection dataset.

HAPT
"""

import os
import urllib
from zipfile import ZipFile

# from . import datasets_dir
import numpy as np

datasets_dir = os.path.dirname(__file__)

import pandas as pd

# Papers
# http://conference.scipy.org/proceedings/scipy2018/pdfs/christian_mcdaniel.pdf
#   github: https://github.com/xtianmcd/accelstm
# https://www.mdpi.com/1424-8220/17/11/2556/htm
# https://arxiv.org/pdf/1801.04503.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.641.2285&rep=rep1&type=pdf


DATASET_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip"
_, DATASET_FILE = os.path.split(DATASET_URL)
DATASET_SUBDIR, _ = os.path.splitext(DATASET_FILE)
DATASET_SUBDIR = DATASET_SUBDIR.replace("%20", "_")
OUTPUT_DIR = os.path.join(datasets_dir, DATASET_SUBDIR)


def download_if_needed():
    """Downloads and extracts .zip if needed. """
    if not os.path.exists(OUTPUT_DIR):
        if not os.path.exists(os.path.join(datasets_dir, DATASET_FILE)):
            print("Downloading .zip file...")
            urllib.request.urlretrieve(
                DATASET_URL, os.path.join(datasets_dir, DATASET_FILE)
            )
        assert os.path.exists(os.path.join(datasets_dir, DATASET_FILE))

        print(f"Extracting to {OUTPUT_DIR}...")
        zip = ZipFile(os.path.join(datasets_dir, DATASET_FILE))
        zip.extractall(OUTPUT_DIR)

    assert os.path.exists(OUTPUT_DIR)


def get_label_series():
    activity_labels_path = os.path.join(OUTPUT_DIR, "activity_labels.txt")
    print(f"Loading file {activity_labels_path}")
    label_series = pd.read_csv(activity_labels_path, sep=r"\s+", header=None)
    label_series.columns = ["class_number", "class_label"]
    label_series = label_series.dropna(how="any", axis=0)
    return pd.Series(
        label_series["class_label"].values, index=label_series["class_number"].values
    )


def load_labels():
    label_file_path = os.path.join(OUTPUT_DIR, "RawData/labels.txt")
    print(f"Reading labels file {label_file_path}")
    df_labels = pd.read_csv(label_file_path, r"\s+", header=None)
    df_labels.columns = [
        "experiment_id",
        "subject_id",
        "activity_id",
        "label_start_point",
        "label_end_point",
    ]

    label_series = get_label_series()

    df_labels["activity_label"] = df_labels["activity_id"].map(label_series)

    return df_labels.sort_values(
        ["experiment_id", "label_start_point", "label_end_point"]
    )


def separate_subjects():
    all_subjects = set(range(1, 31))
    # validation_subjects = {4, 12, 20, 27}  # 3 test (4, 12, 20) + 1 train (27)
    validation_subjects = {5, 16, 27}
    test_subjects = {2, 4, 9, 10, 12, 13, 18, 20, 24}  # Given test subjects
    train_subjects = all_subjects - test_subjects - validation_subjects
    assert train_subjects | test_subjects | validation_subjects == all_subjects
    assert set() == validation_subjects & test_subjects
    assert set() == train_subjects & test_subjects
    assert set() == train_subjects & validation_subjects
    return train_subjects, validation_subjects, test_subjects


def get_column_info():
    sensor_types = ["accel", "gyro"]
    file_names = ["acc", "gyro"]
    num_chans = [3, 3]
    assert len(num_chans) == len(sensor_types)
    types = []
    names = []
    file_prefixes = []
    for i in range(len(sensor_types)):
        t = sensor_types[i]
        nc = num_chans[i]
        types.extend([t] * nc)
        file_prefixes.extend([file_names[i]] * nc)
        names.extend([f"{t[0]}{j + 1}" for j in range(nc)])

    # Note that this segfaults if the input data is a tuple and not a list. WTF pandas?
    df = pd.DataFrame(
        [types, file_prefixes], index=["sensor_type", "file_prefix"], columns=names
    ).T
    df.index.name = "name"
    df["output"] = False

    df = df.reindex(
        df.index.append(
            pd.Index(["experiment_id", "subject_id", "activity_id", "activity_label"])
        )
    )

    df.loc[["activity_id", "activity_label"], "output"] = True

    return df


def get_data():
    train_subjects, validation_subjects, test_subjects = separate_subjects()
    labels = load_labels()
    col_info = get_column_info()

    output_cols = col_info.index[col_info.output == True]
    not_input_cols = col_info.index[~(col_info.output == False)]
    all_sensors = col_info.loc[col_info.output == False, "sensor_type"].unique()
    train_data = []
    validation_data = []
    test_data = []
    for exp_num, exp_labels in labels.groupby("experiment_id"):
        user_id = exp_labels["subject_id"].unique()  # Should only be 1
        assert len(user_id) == 1
        user_id = user_id[0]

        exp_data = []
        for t in all_sensors:
            sensor_type_mask = t == col_info.sensor_type
            prefix = col_info.loc[sensor_type_mask, "file_prefix"].unique()[0]
            file_path = os.path.join(
                OUTPUT_DIR,
                "RawData",
                f"{prefix}_exp{exp_num:02d}_user{user_id:02d}.txt",
            )
            print(f"Reading file {file_path}")
            tmp_df = pd.read_csv(file_path, sep=r"\s+", header=None)
            tmp_df.columns = col_info.index[sensor_type_mask]
            exp_data.append(tmp_df)
        exp_data = pd.concat(exp_data, axis=1, ignore_index=False)

        # Make 1 based to match everything else
        exp_data.index = exp_data.index + 1
        exp_data = pd.concat(
            (exp_data, pd.DataFrame(index=exp_data.index, columns=not_input_cols)),
            axis=1,
            ignore_index=False,
        )

        for _, l in exp_labels.iterrows():
            exp_data.loc[
                l["label_start_point"] : l["label_end_point"], output_cols
            ] = l[output_cols].values

        # Just to help make sure matches expectations
        exp_data = exp_data.reset_index(drop=True)

        exp_data["experiment_id"] = exp_num
        exp_data["subject_id"] = user_id
        exp_data["activity_id"] = exp_data["activity_id"].fillna(0)
        exp_data["activity_label"] = "UNKNOWN"

        if user_id in train_subjects:
            train_data.append(exp_data)
        elif user_id in validation_subjects:
            validation_data.append(exp_data)
        elif user_id in test_subjects:
            test_data.append(exp_data)
        else:
            raise ValueError(f"Unexpected subject {user_id}")

    train_data = pd.concat(train_data)
    validation_data = pd.concat(validation_data)
    test_data = pd.concat(test_data)
    return train_data, validation_data, test_data


def get_dfs_processed():
    df_train, df_val, df_test = get_data()
    col_df = get_column_info()
    # Calculate feature normalizing stats from training set only
    norm_mean = df_train.mean(axis=0)
    norm_std = df_train.std(axis=0)

    # Apply to all sets to normalize features
    input_features = col_df.index[col_df.output == False]
    for df in (df_train, df_val, df_test):
        # de-mean
        df.loc[:, input_features] -= norm_mean

        # unit std dev
        df.loc[:, input_features] /= norm_std

        # interpolate and NA's -> 0
        df.loc[:, input_features] = df.loc[:, input_features].interpolate().fillna(0)

    return df_train, df_val, df_test


_df_dicts = {}


def get_or_make_dfs():
    if _df_dicts:
        return _df_dicts
    download_if_needed()

    cache_dir = os.path.join(OUTPUT_DIR, "cache")
    if not os.path.isdir(cache_dir) or not os.path.isfile(
        os.path.join(cache_dir, "df_train.df.pkl")
    ):
        print("Smartphone data not cached. Creating cache now...")
        try:
            os.makedirs(cache_dir)
        except FileExistsError:
            pass

        df_train, df_val, df_test = get_dfs_processed()
        df_cols = get_column_info()

        s_labels = get_label_series()

        df_train.to_pickle(os.path.join(cache_dir, "df_train.df.pkl"))
        _df_dicts["df_train"] = df_train
        df_val.to_pickle(os.path.join(cache_dir, "df_val.df.pkl"))
        _df_dicts["df_val"] = df_val
        df_test.to_pickle(os.path.join(cache_dir, "df_test.df.pkl"))
        _df_dicts["df_test"] = df_test

        df_cols.to_pickle(os.path.join(cache_dir, "df_cols.df.pkl"))
        _df_dicts["df_cols"] = df_cols
        s_labels.to_pickle(os.path.join(cache_dir, "s_labels.s.pkl"))
        _df_dicts["s_labels"] = s_labels
        print("Caching done.")
    else:
        print("Loading cached data.")
        _df_dicts["df_train"] = pd.read_pickle(
            os.path.join(cache_dir, "df_train.df.pkl")
        )
        _df_dicts["df_val"] = pd.read_pickle(os.path.join(cache_dir, "df_val.df.pkl"))
        _df_dicts["df_test"] = pd.read_pickle(os.path.join(cache_dir, "df_test.df.pkl"))

        _df_dicts["df_cols"] = pd.read_pickle(os.path.join(cache_dir, "df_cols.df.pkl"))
        _df_dicts["s_labels"] = pd.read_pickle(
            os.path.join(cache_dir, "s_labels.s.pkl")
        )
        print("Loaded.")

    return _df_dicts


def get_x_y_contig(
    which_set="train",
    dfs_dict=None,
    y_cols=("y_activity",),
    sensor_subset=None,
    include_transitions=False,
):
    """ Load X and y as contiguous time vectors (with various runs concatenated together).

    Parameters
    ----------
    which_set: str
        which of the pre-defined splits to load. Can specify multiples, separated by '+'. E.g.,
        "train", "val", "test", "train+val", etc.
    dfs_dict: dict
        Can provide pre-laoded df_dict to save time. (deprecated)
    y_cols: List[str]
        List of which label columns to return (e.g., ['y_gesture'], ['y_locomotion'], or ['y_gesture', 'y_locomotion']
    sensor_subset: ty.Iterable
        Which subset of sensors to include. Valid values include:
            "accels", "accel"
            "gyros", "gyro"
            "accels+gyros", "accel+gyro"
            "all"
    include_transitions : bool
    """
    y_cols = list(y_cols)
    if not dfs_dict:
        dfs_dict = get_or_make_dfs()

    assert isinstance(y_cols, list)

    df = []
    for _which_set in which_set.split("+"):
        df.append(dfs_dict["df_" + _which_set])

    df = pd.concat(df)

    df_cols = dfs_dict["df_cols"]

    if not sensor_subset or sensor_subset == "all":
        cols = df_cols.index[df_cols.output == False]
    else:
        cols = []
        for s in sensor_subset.split("+"):
            if s.endswith("s"):
                s = s[:-1]
            cols.append(df_cols.index[df_cols.sensor_type == s])
        cols = pd.concat(cols)

    Xc = df[cols].values
    s_labels = dfs_dict["s_labels"]
    activity_outputs = df["activity_id"].copy()
    if not include_transitions:
        transition_mask = s_labels.str.contains("_TO_")
        transition_activities = s_labels.index[transition_mask]
        activity_outputs[activity_outputs.isin(transition_activities)] = np.nan
        s_labels = s_labels[~transition_mask]

    # Replace nulls with 0
    activity_outputs = activity_outputs.fillna(0)

    ycs = [activity_outputs.values]

    # Include the null class
    output_spec_dict = {
        "name": "y_activity",
        "num_classes": len(s_labels) + 1,
        "classes": [""] + s_labels.sort_index().to_list(),
    }

    data_spec = {
        "dataset_name": "smartphone",
        "input_channels": Xc.shape[1],
        "n_outputs": len(ycs),
        "input_features": cols.to_list(),
        "output_spec": [output_spec_dict],
    }

    return Xc, ycs, data_spec


if __name__ == "__main__":
    pass
