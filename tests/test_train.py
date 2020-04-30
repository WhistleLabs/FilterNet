# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os

import pytest
import torch

from filternet.datasets import opportunity as opp, sliding_window_x_y
from filternet.training.trainable import MPTrainable


@pytest.fixture
def dfs_dict():
    return opp.get_or_make_dfs()


@pytest.fixture
def x_y_dict(dfs_dict):
    wl = 64
    xys = {}
    for which_set in ["train", "val", "test"]:
        Xc, ycs, data_spec = opp.get_x_y_contig(which_set, dfs_dict=dfs_dict)

        X, ys = sliding_window_x_y(Xc, ycs, win_len=wl)

        assert X.shape[1] == 113
        assert X.shape[2] == wl
        for y in ys:
            assert y.shape[1] == wl

        xys["X_" + which_set] = torch.Tensor(X)
        xys["ys_" + which_set] = [torch.Tensor(y).long() for y in ys]
    xys["win_len"] = wl
    return xys


def test_train_val_test_model():
    trainable = MPTrainable(
        {
            "name": "unittest",
            "loss_func": "cross_entropy",
            "decimation": 10,
            "base_config": "base_cnn",
            "model_config": {"scale": (1.0 / 16)},
        }
    )

    trainer = trainable.trainer
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.win_len is not None
    assert trainer.loss_func is not None

    assert trainer.dl_train is not None
    assert trainer.dl_val is not None
    assert trainer.dl_test is not None

    # one training iteration
    ret = trainable.train()
    assert not ret["done"]
    assert ret["training_iteration"] == 1
    assert "train_loss" in ret
    assert "train_acc" in ret
    assert "mean_loss" in ret
    assert "mean_accuracy" in ret
    assert "val_f1" in ret
    assert ret["config"]["loss_func"] == trainer.loss_func
    print(ret)

    trainable.trainer.loss_func = "binary_cross_entropy"
    ret = trainable.train()
    assert not ret["done"]
    assert ret["training_iteration"] == 2
    assert "train_loss" in ret
    print(ret)

    ret = trainable.train()
    assert not ret["done"]
    assert ret["training_iteration"] == 3
    assert "train_loss" in ret
    print(ret)

    trainer.train_state.extra["temp"] = 1

    path = trainable.save()
    assert os.path.exists(path)
    assert trainer.train_state.extra["temp"] == 1
    trainer.train_state.extra["temp"] = 2
    assert trainer.train_state.extra["temp"] == 2
    trainable.restore(path)
    assert trainer.train_state.extra["temp"] == 1  # make sure restoring state worked.
    ret = trainable.train()
    assert ret["training_iteration"] == 4

    print(ret)


#
# def test_train_diff_dimensionalities():
#
#     trainable = train.CNNLSTMTrainable({'output_type': 'many_to_one_takelast', 'decimation': 10, 'loss': 'binary_cross_entropy', 'scale': (1.0 / 8)})
#     ret = trainable.train()
#     assert ret['training_iteration'] == 1
#     assert 'train_loss' in ret
#     assert 'train_acc' in ret
#     assert 'mean_loss' in ret
#     assert 'mean_accuracy' in ret
#     assert 'val_f1' in ret
#
#     trainable = train.CNNLSTMTrainable(
#         {'output_type': 'many_to_one_takelast', 'decimation': 10, 'loss': 'cross_entropy', 'scale': (1.0/8)})
#     ret = trainable.train()
#     assert ret['training_iteration'] == 1
#     assert 'train_loss' in ret
#     assert 'train_acc' in ret
#     assert 'mean_loss' in ret
#     assert 'mean_accuracy' in ret
#     assert 'val_f1' in ret
#
#     trainable = train.CNNLSTMTrainable({'output_type': 'many_to_many', 'decimation': 10, 'scale': (1.0 / 8)})
#     ret = trainable.train()
#     assert ret['training_iteration'] == 1
#     assert 'train_loss' in ret
#     assert 'train_acc' in ret
#     assert 'mean_loss' in ret
#     assert 'mean_accuracy' in ret
#     assert 'val_f1' in ret
#
# # Re-enable when ability to use different models is re-enabled:
# def test_train_diff_models():
#     trainable = train.CNNLSTMTrainable({'model_class': 'DeepConvLSTM', 'decimation': 10, 'scale': (1.0 / 8)})
#     ret = trainable.train()
#     assert ret['training_iteration'] == 1
#     assert 'train_loss' in ret
#     assert 'train_acc' in ret
#     assert 'mean_loss' in ret
#     assert 'mean_accuracy' in ret
#     assert 'val_f1' in ret
#     ret = trainable.test_with_overlap()
#     assert 'test_f1' in ret
#     for o in ret['output_records']:
#         assert 'classification_report_txt' in o
#
#
#     trainable = train.CNNLSTMTrainable({'model_class': 'FilterNet', 'decimation': 10, 'scale': (1.0 / 8)})
#     ret = trainable.train()
#     assert ret['training_iteration'] == 1
#     assert 'train_loss' in ret
#     assert 'train_acc' in ret
#     assert 'mean_loss' in ret
#     assert 'mean_accuracy' in ret
#     assert 'val_f1' in ret
#     ret = trainable.test_with_overlap()
#     assert 'test_f1' in ret
#     for o in ret['output_records']:
#         assert 'classification_report_txt' in o
