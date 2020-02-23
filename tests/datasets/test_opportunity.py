# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os

import pytest

from datasets import opportunity as opp
from datasets import sliding_window_x_y


@pytest.fixture
def dfs_dict():
    return opp.get_or_make_dfs()


def test_download_opportunity():
    opp.download_if_needed()
    assert os.path.exists(os.path.join(opp.datasets_dir, opp.DATASET_SUBDIR))
    assert os.path.exists(os.path.join(opp.datasets_dir, opp.DATASET_FILE))


def test_get_or_make_dfs(dfs_dict):
    assert dfs_dict["df_train"].shape == (497014, 252)
    assert dfs_dict["df_val"].shape == (60949, 252)
    assert dfs_dict["df_test"].shape == (118750, 252)

    for df in [dfs_dict["df_train"], dfs_dict["df_val"], dfs_dict["df_test"]]:
        assert df.isna().sum().sum() == 0

    assert dfs_dict["df_labels_locomotion"].shape == (5, 3)
    assert dfs_dict["df_labels_gestures"].shape == (18, 3)

    assert dfs_dict["df_cols"].shape == (250, 6)


def test_get_x_y(dfs_dict):
    lens = {}
    for which_set in ["train", "train+val", "val", "test"]:
        Xc, ycs, data_spec = opp.get_x_y_contig(which_set, dfs_dict=dfs_dict)
        wl = 128
        X, ys = sliding_window_x_y(Xc, ycs, win_len=wl)

        assert X.shape[1] == 113
        assert X.shape[2] == wl
        for y in ys:
            assert y.shape[1] == wl

        lens[which_set] = len(Xc)
        assert data_spec["input_channels"] == 113
        assert len(data_spec["input_features"]) == data_spec["input_channels"]
        assert data_spec["n_outputs"] == len(data_spec["output_spec"])

        for o in data_spec["output_spec"]:
            assert "name" in o
            assert o["num_classes"] == len(o["classes"])

        assert "dataset_name" in data_spec

    assert lens["train"] + lens["val"] == lens["train+val"]


def test_get_different_outputs(dfs_dict):
    with pytest.raises(AssertionError):
        Xc, ycs, data_spec = opp.get_x_y_contig(dfs_dict=dfs_dict, y_cols="y_gesture")
    Xc, ycs, data_spec = opp.get_x_y_contig(dfs_dict=dfs_dict, y_cols=["y_gesture"])
    assert len(ycs) == 1
    Xc, ycs2, data_spec = opp.get_x_y_contig(
        dfs_dict=dfs_dict, y_cols=["y_gesture", "y_locomotion"]
    )
    assert len(ycs2) == 2
    assert ycs[0].shape == ycs2[0].shape


def test_get_sensor_subsets(dfs_dict):
    lens = {}
    expected_lens = {
        "accels": 15,
        "gyros": 15,
        "accels+gyros": 30,
        "accels+gyros+magnetic": 45,
        "opportunity": 113,
        None: 113,
    }
    for sensor_subset in [
        None,
        "accels",
        "gyros",
        "accels+gyros",
        "accels+gyros+magnetic",
        "opportunity",
    ]:
        Xc, ycs, data_spec = opp.get_x_y_contig(
            "train+val", sensor_subset=sensor_subset, dfs_dict=dfs_dict
        )
        assert Xc.shape[1] == expected_lens[sensor_subset]


def test_urls():
    assert opp.DATASET_FILE == "OpportunityUCIDataset.zip"
    assert opp.DATASET_SUBDIR == "OpportunityUCIDataset"
