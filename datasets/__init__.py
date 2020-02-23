# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os

import numpy as np
import sklearn

datasets_dir = os.path.split(__file__)[0]


def sliding_window_x_y(Xc, ycs, win_len=128, step=None, shuffle=True):
    if step is None:
        step = int(win_len / 2)
    start_idxs = np.arange(0, len(Xc) - win_len, step)
    X = (
        np.array([Xc[i : i + win_len] for i in start_idxs])
        .transpose([0, 2, 1])
        .astype(np.float32)
    )  # [N, C, L]
    ys = [
        np.array([yc[i : i + win_len] for i in start_idxs]).astype(np.long) 
        for yc in ycs
    ]  # [len(ycs), N, L]
    if shuffle:
        X, *ys = sklearn.utils.shuffle(X, *ys, random_state=0)
        return X, ys
    else:
        return X, ys
