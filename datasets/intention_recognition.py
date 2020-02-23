# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

""" EEG Based intention recognition

Good paper to base it off of:
https://arxiv.org/pdf/1708.06578.pdf
https://github.com/pbashivan/EEGLearn
"""

import os
import urllib
from zipfile import ZipFile

# from . import datasets_dir
import numpy as np
import pyedflib as pyedflib

datasets_dir = os.path.dirname(__file__)

import pandas as pd

DATASET_URL = "https://www.physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip"
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


def separate_subjects():
    all_subjects = set(range(1, 110))
    all_subjects -= {89}  # Screwed up according to paper
    # validation_subjects = {4, 12, 20, 27}  # 3 test (4, 12, 20) + 1 train (27)

    np.random.seed(seed=123)
    test_subjects = set(
        np.random.choice(
            list(all_subjects), size=int(len(all_subjects) // 20), replace=False
        )
    )
    validation_subjects = set(
        np.random.choice(
            list(all_subjects - test_subjects),
            size=int((len(all_subjects) - len(test_subjects)) // 20),
            replace=False,
        )
    )
    train_subjects = all_subjects - test_subjects - validation_subjects

    print(f"test_subjects ({len(test_subjects)}): {test_subjects}")
    print(f"validation_subjects ({len(validation_subjects)}): {validation_subjects}")
    print(f"train_subjects ({len(train_subjects)}): {train_subjects}")

    assert train_subjects | test_subjects | validation_subjects == all_subjects
    assert set() == validation_subjects & test_subjects
    assert set() == train_subjects & test_subjects
    assert set() == train_subjects & validation_subjects
    return train_subjects, validation_subjects, test_subjects


def get_label_series():
    return pd.Series(
        {
            0: "Eyes Closed",
            1: "Left Fist",
            2: "Right Fist",
            3: "Both Fists",
            4: "Both Feet",
        }
    )


def get_column_info():
    sensor_types = ["eeg"]
    num_chans = [64]
    assert len(num_chans) == len(sensor_types)
    types = []
    names = []
    file_path = os.path.join(OUTPUT_DIR, "files", "S001", "S001R01.edf")
    f = pyedflib.EdfReader(file_path)
    names = f.getSignalLabels()
    for i in range(len(sensor_types)):
        t = sensor_types[i]
        nc = num_chans[i]
        types.extend([t] * nc)

    # Note that this segfaults if the input data is a tuple and not a list. WTF pandas?
    df = pd.DataFrame([types], index=["sensor_type"], columns=names).T
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
    col_info = get_column_info()

    train_data = []
    validation_data = []
    test_data = []

    left_or_right_trials = {4, 8, 12}
    both_trials = {6, 10, 14}
    label_series = get_label_series()
    all_trials = left_or_right_trials | both_trials
    all_subjects = train_subjects | validation_subjects | test_subjects
    for sub in all_subjects:
        sub_str = f"S{sub:03d}"
        print(f"Loading data for subject {sub}")
        sub_folder = os.path.join(OUTPUT_DIR, "files", sub_str)
        sub_dfs = []

        for t in all_trials:
            # https://pyedflib.readthedocs.io/en/latest/
            file_path = os.path.join(sub_folder, f"{sub_str}R{t:02d}.edf")
            # print(f"Loading file {file_path}")
            f = pyedflib.EdfReader(file_path)

            n = f.signals_in_file
            assert n == 64
            trial_data = np.empty((f.getNSamples()[0], n), order="F")
            for i in np.arange(n):
                trial_data[:, i] = f.readSignal(i)

            output_classes = np.zeros(trial_data.shape[0], order="F")
            start_times, durations, labels = f.readAnnotations()
            sr = 160
            for st, dur, lab in zip(start_times, durations, labels):
                s_ind = int(st * sr)
                e_ind = int(s_ind + dur * sr)
                if lab == "T0":
                    output_classes[s_ind:e_ind] = 0
                elif lab == "T1":
                    output_classes[s_ind:e_ind] = 1 if t in left_or_right_trials else 3
                elif lab == "T2":
                    output_classes[s_ind:e_ind] = 2 if t in left_or_right_trials else 4
                else:
                    raise ValueError(f"The label {lab} is not defined")

            df = pd.DataFrame(
                trial_data, columns=col_info.index[col_info.output == False]
            )
            df["experiment_id"] = t
            df["subject_id"] = sub
            df["activity_id"] = output_classes
            df["activity_label"] = df["activity_id"].map(label_series)
            sub_dfs.append(df)

        if sub in train_subjects:
            train_data.extend(sub_dfs)
        elif sub in validation_subjects:
            validation_data.extend(sub_dfs)
        elif sub in test_subjects:
            test_data.extend(sub_dfs)
        else:
            raise ValueError(f"Unexpected subject {sub}")

        print(f"Done with subject {sub}")

    print("Creating training DF")
    train_data = pd.concat(train_data, axis=0, ignore_index=True)
    print("Creating validation DF")
    validation_data = pd.concat(validation_data, axis=0, ignore_index=True)
    print("Creating test DF")
    test_data = pd.concat(test_data, axis=0, ignore_index=True)
    print("Created all data frames")
    return train_data, validation_data, test_data


def get_dfs_processed():
    df_train, df_val, df_test = get_data()
    col_df = get_column_info()

    # Apply to all sets to normalize features
    input_features = col_df.index[col_df.output == False]

    # Calculate feature normalizing stats from training set only
    norm_mean = df_train.loc[:, input_features].mean(axis=0)
    norm_std = df_train.loc[:, input_features].std(axis=0)

    print("Normalizing dataframes")
    for df in (df_test, df_val, df_train):
        # de-mean
        df.loc[:, input_features] = df.loc[:, input_features] - norm_mean

        # unit std dev
        df.loc[:, input_features] = df.loc[:, input_features].divide(norm_std, axis=1)

        # interpolate and NA's -> 0
        df.loc[:, input_features] = df.loc[:, input_features].interpolate().fillna(0)

    print("Done normalizing dataframes")
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
        print("Intention recognition data not cached. Creating cache now...")
        try:
            os.makedirs(cache_dir)
        except FileExistsError:
            pass

        df_train, df_val, df_test = get_dfs_processed()
        df_cols = get_column_info()

        s_labels = get_label_series()

        print("Saving data to cache")
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
    which_set="train", dfs_dict=None, y_cols=["y_activity"], sensor_subset=None
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
    """
    if not dfs_dict:
        dfs_dict = get_or_make_dfs()

    assert type(y_cols) == list

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

    # Replace nulls with 0
    activity_outputs = activity_outputs.fillna(0)

    ycs = [activity_outputs.values]

    output_spec_dict = {
        "name": "y_activity",
        "num_classes": len(s_labels),
        "classes": s_labels.sort_index().to_list(),
    }

    data_spec = {
        "dataset_name": "intention",
        "input_channels": Xc.shape[1],
        "n_outputs": len(ycs),
        "input_features": cols.to_list(),
        "output_spec": [output_spec_dict],
    }

    return Xc, ycs, data_spec


if __name__ == "__main__":
    pass
    # get_x_y_contig()
