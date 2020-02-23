# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

""" Loads the opportunity dataset. Based largely on
https://github.com/sussexwearlab/DeepConvLSTM/blob/master/preprocess_data.py
but refactored and with a slightly different normalization

(we de-mean and normalize to unit std deviation using
the statistics of the training set, instead of rescaling + clipping from 0->1 with predefined limits as
the referenced repo does. )

"""

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from zipfile import ZipFile

import pandas as pd

from . import datasets_dir

# datasets_dir = os.path.dirname(__file__)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
_, DATASET_FILE = os.path.split(DATASET_URL)
DATASET_SUBDIR, _ = os.path.splitext(DATASET_FILE)

# +
fns_train = [
    "dataset/S1-ADL4.dat",
    "dataset/S1-Drill.dat",
    "dataset/S1-ADL5.dat",
    "dataset/S1-ADL1.dat",
    "dataset/S1-ADL2.dat",
    "dataset/S1-ADL3.dat",
    "dataset/S2-ADL2.dat",
    "dataset/S3-ADL2.dat",
    "dataset/S3-ADL1.dat",
    "dataset/S2-ADL1.dat",
    "dataset/S3-Drill.dat",
    #  'dataset/S4-ADL4.dat',
    #  'dataset/S4-ADL5.dat',
    "dataset/S2-Drill.dat",
    #  'dataset/S4-ADL2.dat',
    #  'dataset/S4-ADL3.dat',
    #  'dataset/S4-ADL1.dat',
    #  'dataset/S4-Drill.dat'
]

fns_val = ["dataset/S2-ADL3.dat", "dataset/S3-ADL3.dat"]

fns_test = [
    "dataset/S2-ADL4.dat",
    "dataset/S2-ADL5.dat",
    "dataset/S3-ADL4.dat",
    "dataset/S3-ADL5.dat",
]


def make_df_cols():
    """ Make a dataframe of the columns in the datafiles that can be downselected for various sensor subsets, lablels, etc."""

    # Read in columns list
    with open(
        os.path.join(datasets_dir, DATASET_SUBDIR, "dataset/column_names.txt")
    ) as f:
        txt = f.read()

    # Extract Feature columns first
    pattern = "^Column: (?P<col_no_matlab>\d*) (?P<cat>\S*\s?\S*) (?P<posn>\S*) (?P<chan>\S*); .*unit = (?P<unit>.*)$"

    df_feat_cols = pd.DataFrame.from_records(
        [m.groupdict() for m in re.compile(pattern, re.MULTILINE).finditer(txt)]
    )
    df_feat_cols["col_no"] = df_feat_cols["col_no_matlab"].astype(int) - 1
    df_feat_cols = df_feat_cols.set_index("col_no").sort_index()
    df_feat_cols["name"] = (
        (
            df_feat_cols.cat.str.slice(0, 2)
            + " "
            + df_feat_cols["posn"]
            + " "
            + df_feat_cols["chan"]
        )
        .str.replace("^", "hi")
        .str.replace("_", "lo")
    )

    # Then label columns
    pattern = "^Column: (?P<col_no_matlab>\d*) (?P<name>\S*\s?\S*)$"
    df_label_cols = pd.DataFrame.from_records(
        [m.groupdict() for m in re.compile(pattern, re.MULTILINE).finditer(txt)]
    )
    df_label_cols["col_no"] = df_label_cols["col_no_matlab"].astype(int) - 1
    df_label_cols = df_label_cols.set_index("col_no").sort_index()

    # Combine to get all columns
    df_cols = pd.concat([df_feat_cols, df_label_cols], sort=True).sort_index()

    return df_cols


def make_df_labels():
    """ Make a dataframe of the different class labels """
    # +
    pattern = "^(?P<src_idx>\d*)   -   (?P<track_name>\S*)   -   (?P<label_name>.*)$"
    with open(
        os.path.join(datasets_dir, DATASET_SUBDIR, "dataset/label_legend.txt")
    ) as f:
        txt = f.read()
    df_label_legend = pd.DataFrame.from_records(
        [m.groupdict() for m in re.compile(pattern, re.MULTILINE).finditer(txt)]
    )

    # Label mapping for locomotion
    df_labels_locomotion = df_label_legend.query(
        'track_name == "Locomotion"'
    ).reset_index(drop=True)
    df_labels_locomotion["idx"] = df_labels_locomotion.index + 1
    df_labels_locomotion.src_idx = df_labels_locomotion.src_idx.astype(int)
    df_labels_locomotion = df_labels_locomotion.set_index("src_idx", drop=True)
    df_labels_locomotion.loc[0] = ("Null", "", 0)

    # Label mapping for gestures
    df_labels_gesture = df_label_legend.query(
        'track_name == "ML_Both_Arms"'
    ).reset_index(drop=True)
    df_labels_gesture["idx"] = df_labels_gesture.index + 1
    df_labels_gesture.src_idx = df_labels_gesture.src_idx.astype(int)
    df_labels_gesture = df_labels_gesture.set_index("src_idx", drop=True)
    df_labels_gesture.loc[0] = ("Null", "", 0)

    return df_labels_locomotion, df_labels_gesture


def download_if_needed():
    """Downloads and extracts .zip if needed. """

    if not os.path.exists(os.path.join(datasets_dir, DATASET_SUBDIR)):
        if not os.path.exists(os.path.join(datasets_dir, DATASET_FILE)):
            print("Downloading .zip file...")
            urllib.request.urlretrieve(
                DATASET_URL, os.path.join(datasets_dir, DATASET_FILE)
            )
        assert os.path.exists(os.path.join(datasets_dir, DATASET_FILE))

        print("Extracting...")
        zip = ZipFile(os.path.join(datasets_dir, DATASET_FILE))
        zip.extractall(datasets_dir)

    assert os.path.exists(os.path.join(datasets_dir, DATASET_SUBDIR))


def load_opp_dataset(fn):
    """ Load an individual file """
    print(fn)
    return pd.read_csv(
        os.path.join(datasets_dir, DATASET_SUBDIR, fn), header=None, sep="\s+"
    )


def get_dfs_raw():
    """ Load dataframes of concatenated data files for the three pre-defined splits. """
    df_train = pd.concat([load_opp_dataset(fn) for fn in fns_train])
    df_val = pd.concat([load_opp_dataset(fn) for fn in fns_val])
    df_test = pd.concat([load_opp_dataset(fn) for fn in fns_test])

    return df_train, df_val, df_test


def get_dfs_processed():
    """ Load and preprocess the data from raw data down to more manageable dataframes that can be chopped
    up a bit for specific purposes. This includes de-meaning, etc."""
    df_cols = make_df_cols()
    df_feat_cols = df_cols[~df_cols.cat.isna()]

    df_labels_locomotion, df_labels_gestures = make_df_labels()

    df_train, df_val, df_test = get_dfs_raw()

    for df in [df_train, df_val, df_test]:
        # Meaningful column names
        df.columns = df_cols.name.values

    # Calculate feature normalizing stats from training set only
    norm_mean = df_train.loc[:, df_feat_cols.name].mean()
    norm_std = df_train.loc[:, df_feat_cols.name].std()

    # Apply to all sets to normalize features
    for df in [df_train, df_val, df_test]:
        # de-mean
        df.loc[:, df_feat_cols.name] -= norm_mean

        # unit std dev
        df.loc[:, df_feat_cols.name] /= norm_std

        # interpolate and NA's -> 0
        df.loc[:, df_feat_cols.name] = (
            df.loc[:, df_feat_cols.name].interpolate().fillna(0)
        )

        df["y_locomotion"] = df["Locomotion"].map(df_labels_locomotion.idx)
        df["y_gesture"] = df["ML_Both_Arms"].map(df_labels_gestures.idx)

    return df_train, df_val, df_test


_df_dicts = None


def get_or_make_dfs():
    """ Loads pre-processed dataframes from disk, if they exist; creates them if not. Once loaded, they are
    cached in-memory for fast subsequent loads. """

    global _df_dicts
    if _df_dicts is not None:
        return _df_dicts
    download_if_needed()
    cache_dir = os.path.join(datasets_dir, DATASET_SUBDIR, "cache")
    if not os.path.exists(cache_dir):
        print("Opportunity data not cached. Creating cache now...")
        os.makedirs(cache_dir)

        df_cols = make_df_cols()
        df_labels_locomotion, df_labels_gestures = make_df_labels()
        df_train, df_val, df_test = get_dfs_processed()

        df_train.to_pickle(os.path.join(cache_dir, "df_train.df.pkl"))
        df_val.to_pickle(os.path.join(cache_dir, "df_val.df.pkl"))
        df_test.to_pickle(os.path.join(cache_dir, "df_test.df.pkl"))

        df_labels_locomotion.to_pickle(
            os.path.join(cache_dir, "df_labels_locomotion.df.pkl")
        )
        df_labels_gestures.to_pickle(
            os.path.join(cache_dir, "df_labels_gestures.df.pkl")
        )
        df_cols.to_pickle(os.path.join(cache_dir, "df_cols.df.pkl"))
        print("Caching done.")

    print("Loading cached data.")
    df_train = pd.read_pickle(os.path.join(cache_dir, "df_train.df.pkl"))
    df_val = pd.read_pickle(os.path.join(cache_dir, "df_val.df.pkl"))
    df_test = pd.read_pickle(os.path.join(cache_dir, "df_test.df.pkl"))

    df_labels_locomotion = pd.read_pickle(
        os.path.join(cache_dir, "df_labels_locomotion.df.pkl")
    )
    df_labels_gestures = pd.read_pickle(
        os.path.join(cache_dir, "df_labels_gestures.df.pkl")
    )
    df_cols = pd.read_pickle(os.path.join(cache_dir, "df_cols.df.pkl"))
    print("Loaded.")

    _df_dicts = dict(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        df_labels_locomotion=df_labels_locomotion,
        df_labels_gestures=df_labels_gestures,
        df_cols=df_cols,
    )

    return _df_dicts


def get_x_y_contig(
    which_set="train", dfs_dict=None, y_cols=["y_gesture"], sensor_subset=None
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
    sensor_subset: str
        Which subset of sensors to include. Valid values include:
            "accels",
            "gyros",
            "accels+gyros",
            "accels+gyros+magnetic",  << above categories are for motion jacket only
            "opportunity"  <<< all 113 sensors used in opportunity challenge
    """
    if dfs_dict is None:
        dfs_dict = get_or_make_dfs()

    assert type(y_cols) == list

    df = []
    for _which_set in which_set.split("+"):
        df.append(dfs_dict["df_" + _which_set])

    df = pd.concat(df)

    df_cols = dfs_dict["df_cols"]

    # Used for sensor subsets (arm + back IMUs)
    core_posns = ["RUA", "LUA", "RLA", "LLA", "BACK"]

    # For full Opportunity complement of sensors:
    xtra_posns = [
        "L-SHOE",
        "R-SHOE",
        "RKN_",
        "RUA_",
        "RWR",
        "LUA_",
        "HIP",
        "RUA^",
        "RKN^",
        "LWR",
        "LH",
        "LUA^",
        "RH",
    ]

    if sensor_subset is None or sensor_subset == "opportunity":
        posns = core_posns + xtra_posns
        chan_pattern = ""
        cat_pattern = ""
    else:
        posns = core_posns
        # regex for selecting sensors by prefixes
        chan_pattern = "|".join([f"^{s[:3]}" for s in sensor_subset.split("+")])
        cat_pattern = "^Inertial"

    feature_cols = df_cols[
        (
            df_cols.posn.isin(posns)
            & ~(df_cols.chan.str.match("^Quat") == True)
            & (df_cols.chan.str.match(chan_pattern) == True)
            & (df_cols.cat.str.match(cat_pattern) == True)
        )
    ]

    Xc = df[feature_cols.name].values
    ycs = [df[y_col].values for y_col in y_cols]

    output_spec_dict = {
        "y_gesture": {
            "name": "gesture",
            "num_classes": len(dfs_dict["df_labels_gestures"]),
            "classes": dfs_dict["df_labels_gestures"]
            .set_index("idx")
            .sort_index()
            .label_name.to_list(),
        },
        "y_locomotion": {
            "name": "locomotion",
            "num_classes": len(dfs_dict["df_labels_locomotion"]),
            "classes": dfs_dict["df_labels_locomotion"]
            .set_index("idx")
            .sort_index()
            .label_name.to_list(),
        },
    }

    data_spec = {
        "dataset_name": "opportunity",
        "input_channels": Xc.shape[1],
        "n_outputs": len(y_cols),
        "input_features": feature_cols.name.to_list(),
        "output_spec": [output_spec_dict[y_col] for y_col in y_cols],
    }

    return Xc, ycs, data_spec


if __name__ == "__main__":
    pass
