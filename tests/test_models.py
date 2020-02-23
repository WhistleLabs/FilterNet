# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import pytest
import torch

import modeling_paper.models as mo
from datasets import opportunity as opp, sliding_window_x_y


@pytest.fixture
def dfs_dict():
    return opp.get_or_make_dfs()


@pytest.fixture
def x_y_dict():
    wl = 64
    xys = {}
    for which_set in ["train", "val", "test"]:
        Xc, ycs, data_spec = opp.get_x_y_contig(which_set)

        X, ys = sliding_window_x_y(Xc, ycs, win_len=wl)

        assert X.shape[1] == 113
        assert X.shape[2] == wl
        assert ys[0].shape[1] == wl

        xys["X_" + which_set] = torch.Tensor(X)
        xys["ys_" + which_set] = [torch.Tensor(y).long() for y in ys]
    xys["win_len"] = wl
    return xys


def test_make_model():
    net = mo.DeepConvLSTM(scale=0.25)


def test_transform_output_m2o(x_y_dict):
    net = mo.DeepConvLSTM(output_type="many_to_one_takelast", scale=(1.0 / 8))
    N = 10
    X = x_y_dict["X_train"][:N]
    ys = [y[:N] for y in x_y_dict["ys_train"]]

    y_outs = net(X)
    for y_out, num_output_classes in zip(y_outs, net.num_output_classes):
        assert y_out.shape == (N, num_output_classes, 1)
    y_comps = net.transform_targets(ys)
    for y_comp, y_out in zip(y_comps, y_outs):
        assert y_comp.shape == y_out.shape


def test_transform_output_m2m(x_y_dict):
    net = mo.DeepConvLSTM(output_type="many_to_many", scale=(1.0 / 8))
    N = 10
    X = x_y_dict["X_train"][:N]
    ys = [y[:N] for y in x_y_dict["ys_train"]]

    y_outs = net(X)
    for y_out, num_output_classes in zip(y_outs, net.num_output_classes):
        assert y_out.shape == (
            N,
            num_output_classes,
            x_y_dict["win_len"] - 2 * net.padding_lost_per_side,
        )
    y_comps = net.transform_targets(ys)
    for y_comp, y_out in zip(y_comps, y_outs):
        assert y_comp.shape == y_out.shape


@pytest.mark.skip("No need to generate 10,000 different models every time!")
def test_make_cnn_lstm_models(x_y_dict):
    N = 10
    X = x_y_dict["X_train"][:N]
    ys = [y[:N] for y in x_y_dict["ys_train"]]

    i = 0
    for n_pre in [0, 1, 2]:
        print("n_pre ", n_pre)
        for n_strided in [0, 3]:
            print("n_strided ", n_strided)
            for n_interp in [0, 1, 3]:
                print("n_interp ", n_interp)
                for n_dense_pre_l in [0, 1, 2]:
                    print("n_dense_pre_l", n_dense_pre_l)
                    for n_l in [0, 1, 2]:
                        print("n_l ", n_l)
                        for n_dense_post_l in [0, 1, 2]:
                            print("n_dense_post_l ", n_dense_post_l)
                            for do_pool in [True, False]:
                                print("do_pool ", do_pool)
                                for stride_pos in ["pre", "post"]:
                                    print("stride_pos ", stride_pos)
                                    for dropout in [0, 0.5]:
                                        print("dropout ", dropout)
                                        for bn_pre in [True, False]:
                                            print("bn_pre ", bn_pre)
                                            i += 1
                                            opts = dict(
                                                output_type="many_to_many",
                                                scale=(1.0 / 4),
                                                n_pre=n_pre,
                                                n_strided=n_strided,
                                                n_interp=n_interp,
                                                n_dense_pre_l=n_dense_pre_l,
                                                n_l=n_l,
                                                n_dense_post_l=n_dense_post_l,
                                                do_pool=do_pool,
                                                stride_pos=stride_pos,
                                                dropout=dropout,
                                                bn_pre=bn_pre,
                                            )
                                            print(i)
                                            try:
                                                net = mo.FilterNet(**opts)
                                                y_outs = net(X)
                                                for y_out, num_output_classes in zip(
                                                    y_outs, net.num_output_classes
                                                ):
                                                    assert y_out.shape == (
                                                        N,
                                                        num_output_classes,
                                                        x_y_dict["win_len"]
                                                        / net.output_stride,
                                                    )
                                                y_comps = net.transform_targets(ys)
                                                for y_comp, y_out in zip(
                                                    y_comps, y_outs
                                                ):
                                                    assert y_comp.shape == y_out.shape
                                            except Exception as e:
                                                print(opts)
                                                raise
