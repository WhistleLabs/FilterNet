# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os
import pickle
import typing as ty

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import traits.api as t
from torch.utils.data import DataLoader, TensorDataset

from filternet import models
from filternet.training.evalmodel import EvalModel
from filternet.training.train import Trainer


class EnsembleTrainer(t.HasStrictTraits):
    def __init__(self, config={}, **kwargs):
        trainer_template = Trainer(**config)
        super().__init__(trainer_template=trainer_template, config=config, **kwargs)

    config: dict = t.Dict()

    trainer_template: Trainer = t.Instance(Trainer)
    trainers: ty.List[Trainer] = t.List(t.Instance(Trainer))

    n_folds = t.Int(5)

    dl_test: DataLoader = t.DelegatesTo("trainer_template")
    data_spec: dict = t.DelegatesTo("trainer_template")
    cuda: bool = t.DelegatesTo("trainer_template")
    device: str = t.DelegatesTo("trainer_template")
    loss_func: str = t.DelegatesTo("trainer_template")
    batch_size: int = t.DelegatesTo("trainer_template")
    win_len: int = t.DelegatesTo("trainer_template")
    has_null_class: bool = t.DelegatesTo("trainer_template")
    predict_null_class: bool = t.DelegatesTo("trainer_template")
    name: str = t.Str()

    def _name_default(self):
        import time

        modelstr = "Ensemble"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return f"{modelstr}_{timestr}"

    X_folds = t.Tuple(transient=True)
    ys_folds = t.Tuple(transient=True)

    def _trainers_default(self):
        # Temp trainer for grabbing datasets, etc
        tt = self.trainer_template
        tt.init_data()

        # Combine official train & val sets
        X = torch.cat([tt.dl_train.dataset.tensors[0], tt.dl_val.dataset.tensors[0]])
        ys = [
            torch.cat([yt, yv])
            for yt, yv in zip(
                tt.dl_train.dataset.tensors[1:], tt.dl_val.dataset.tensors[1:]
            )
        ]
        # make folds
        fold_len = int(np.ceil(len(X) / self.n_folds))
        self.X_folds = torch.split(X, fold_len)
        self.ys_folds = [torch.split(y, fold_len) for y in ys]

        trainers = []
        for i_val_fold in range(self.n_folds):
            trainer = Trainer(
                validation_fold=i_val_fold,
                name=f"{self.name}/{i_val_fold}",
                **self.config,
            )

            trainer.dl_test = tt.dl_test

            trainers.append(trainer)

        return trainers

    model: models.BaseNet = t.Instance(torch.nn.Module, transient=True)

    def _model_default(self):
        model = models.FilterNetEnsemble()
        model.set_models([trainer.model for trainer in self.trainers])
        return model

    model_path: str = t.Str()

    def _model_path_default(self):
        return f"saved_models/{self.name}/"

    def init_data(self):
        # Initiate loading of datasets, model
        pass
        # for trainer in self.trainers:
        #     trainer.init_data()

    def init_train(self):
        pass
        # for trainer in self.trainers:
        #     trainer.init_train()

    def train(self, max_epochs=50):
        """ A pretty standard training loop, constrained to stop in `max_epochs` but may stop early if our
        custom stopping metric does not improve for `self.patience` epochs. Always checkpoints
        when a new best stopping_metric is achieved. An alternative to using
        ray.tune for training."""

        for trainer in self.trainers:
            # Add data to trainer

            X_train = torch.cat(
                [
                    arr
                    for i, arr in enumerate(self.X_folds)
                    if i != trainer.validation_fold
                ]
            )
            ys_train = [
                torch.cat(
                    [arr for i, arr in enumerate(y) if i != trainer.validation_fold]
                )
                for y in self.ys_folds
            ]

            X_val = torch.cat(
                [
                    arr
                    for i, arr in enumerate(self.X_folds)
                    if i == trainer.validation_fold
                ]
            )
            ys_val = [
                torch.cat(
                    [arr for i, arr in enumerate(y) if i == trainer.validation_fold]
                )
                for y in self.ys_folds
            ]

            trainer.dl_train = DataLoader(
                TensorDataset(torch.Tensor(X_train), *ys_train),
                batch_size=trainer.batch_size,
                shuffle=True,
            )
            trainer.data_spec = self.trainer_template.data_spec
            trainer.epoch_iters = self.trainer_template.epoch_iters
            trainer.dl_val = DataLoader(
                TensorDataset(torch.Tensor(X_val), *ys_val),
                batch_size=trainer.batch_size,
                shuffle=False,
            )

            # Now clear local vars to save ranm
            X_train = ys_train = X_val = ys_val = None

            trainer.init_data()
            trainer.init_train()
            trainer.train(max_epochs=max_epochs)

            # Clear trainer train and val datasets to save ram
            trainer.dl_train = t.Undefined
            trainer.dl_val = t.Undefined

            print(f"RESTORING TO best model")
            trainer._restore()
            trainer._save()

            trainer.print_train_summary()

            em = EvalModel(trainer=trainer)

            em.run_test_set()
            em.calc_metrics()
            em.calc_ward_metrics()
            print(em.classification_report_df.to_string(float_format="%.3f"))
            em._save()

    def print_train_summary(self):
        for trainer in self.trainers:
            trainer.print_train_summary()

    def _save(self, checkpoint_dir=None, save_model=True, save_trainer=True):
        """ Saves/checkpoints model state and training state to disk. """
        if checkpoint_dir is None:
            checkpoint_dir = self.model_path
        else:
            self.model_path = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # save model params
        model_path = os.path.join(checkpoint_dir, "model.pth")
        trainer_path = os.path.join(checkpoint_dir, "trainer.pth")

        if save_model:
            torch.save(self.model.state_dict(), model_path)
        if save_trainer:
            with open(trainer_path, "wb") as f:
                pickle.dump(self, f)

        return checkpoint_dir

    def _restore(self, checkpoint_dir=None):
        """ Restores model state and training state from disk. """

        if checkpoint_dir is None:
            checkpoint_dir = self.model_path

        model_path = os.path.join(checkpoint_dir, "model.pth")
        trainer_path = os.path.join(checkpoint_dir, "trainer.pth")

        # Reconstitute old trainer and copy state to this trainer.
        with open(trainer_path, "rb") as f:
            other_trainer = pickle.load(f)

        self.__setstate__(other_trainer.__getstate__())

        # Load sub-models
        for trainer in self.trainers:
            trainer._restore()

        # Load model (after loading state in case we need to re-initialize model from config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.model = self.model._model_default()
