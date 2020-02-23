# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os

import pytest

from datasets import sliding_window_x_y
from datasets import smartphone_hapt as ds


@pytest.fixture
def dfs_dict():
    return ds.get_or_make_dfs()


def test_download():
    ds.download_if_needed()
    assert os.path.exists(os.path.join(ds.datasets_dir, ds.DATASET_SUBDIR))
    assert os.path.exists(os.path.join(ds.datasets_dir, ds.DATASET_FILE))


def test_get_or_make_dfs(dfs_dict):
    assert dfs_dict["df_train"].shape == (686203, 10)
    assert dfs_dict["df_val"].shape == (111281, 10)
    assert dfs_dict["df_test"].shape == (325288, 10)

    for df in [dfs_dict["df_train"], dfs_dict["df_val"], dfs_dict["df_test"]]:
        assert not df.isna().any().any()

    assert dfs_dict["s_labels"].shape == (12,)
    assert dfs_dict["df_cols"].shape == (10, 3)


def test_get_x_y(dfs_dict):
    lens = {}
    for which_set in ["train", "train+val", "val", "test"]:
        Xc, ycs, data_spec = ds.get_x_y_contig(which_set, dfs_dict=dfs_dict)
        wl = 128
        X, ys = sliding_window_x_y(Xc, ycs, win_len=wl)

        assert X.shape[1] == data_spec["input_channels"]
        assert X.shape[2] == wl
        for y in ys:
            assert y.shape[1] == wl

        lens[which_set] = len(Xc)
        assert len(data_spec["input_features"]) == data_spec["input_channels"]
        assert data_spec["n_outputs"] == len(data_spec["output_spec"])

        for o in data_spec["output_spec"]:
            assert "name" in o
            assert o["num_classes"] == len(o["classes"])

        assert "dataset_name" in data_spec

    assert lens["train"] + lens["val"] == lens["train+val"]


def test_urls():
    assert ds.DATASET_FILE == "HAPT%20Data%20Set.zip"
    assert ds.DATASET_SUBDIR == "HAPT_Data_Set"
