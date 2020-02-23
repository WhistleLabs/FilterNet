# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

""" Reference architectures """
from copy import deepcopy

ref_archs = {
    "base_cnn": {
        "n_pre": 1,
        "n_strided": 3,
        "n_interp": 0,
        "n_dense_pre_l": 0,
        "n_l": 0,
        "n_dense_post_l": 0,
    },
    "multi_scale_cnn": {
        "n_pre": 1,
        "n_strided": 3,
        "n_interp": 4,
        "n_dense_pre_l": 1,
        "n_l": 0,
        "n_dense_post_l": 0,
    },
    "base_lstm": {
        "n_pre": 0,
        "lr_decay": 1.0,
        "n_strided": 0,
        "n_interp": 0,
        "n_dense_pre_l": 0,
        "n_l": 1,
        "n_dense_post_l": 0,
    },
    "cnn_lstm": {
        "n_pre": 1,
        "n_strided": 3,
        "n_interp": 0,
        "n_dense_pre_l": 1,
        "n_l": 1,
        "n_dense_post_l": 0,
    },
    "multi_scale_cnn_lstm": {
        "n_pre": 1,
        "n_strided": 3,
        "n_interp": 4,
        "n_dense_pre_l": 1,
        "n_l": 1,
        "n_dense_post_l": 0,
    },
}


def get_ref_arch(name):
    return deepcopy(ref_archs[name])
