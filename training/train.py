# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os
import pickle
import typing as ty

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.optim
import traits.api as t
from torch.utils.data import DataLoader, TensorDataset

import models
from datasets import sliding_window_x_y
from models.reference_architectures import get_ref_arch
from mputil import Timer


class EpochMetrics(t.HasStrictTraits):
    f1: float = t.Float()
    loss: float = t.Float()
    acc: float = t.Float()


class EpochRecord(t.HasStrictTraits):
    epoch: int = t.Int()
    train: EpochMetrics = t.Instance(EpochMetrics)
    val: EpochMetrics = t.Instance(EpochMetrics)

    lr: float = t.Float()
    iter_s_cpu: float = t.Float()
    iter_s_wall: float = t.Float()
    should_checkpoint: bool = t.Bool(False)
    done: bool = t.Bool(False)
    stopping_metric: float = t.Float()

    def to_dict(self):
        d = {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and not type(v) == EpochMetrics
        }
        for f in ["train", "val"]:
            em = getattr(self, f)
            if em:
                for k, v in em.__dict__.items():
                    if v is not None:
                        d[f"{f}_{k}"] = v
        return d


class TrainState(t.HasStrictTraits):
    epoch_records: ty.List[EpochRecord] = t.List(EpochRecord, [])
    best_sm: float = t.Float()
    best_loss: float = t.Float()
    best_f1: float = t.Float()
    extra: dict = t.Dict()

    def to_df(self):
        return (
            pd.DataFrame.from_records([er.to_dict() for er in self.epoch_records])
            .set_index("epoch")
            .sort_index(axis=1)
        )


class Trainer(t.HasStrictTraits):
    model: models.BaseNet = t.Instance(torch.nn.Module, transient=True)

    def _model_default(self):

        # Merge 'base config' (if requested) and any overrides in 'model_config'
        if self.base_config:
            model_config = get_ref_arch(self.base_config)
        else:
            model_config = {}
        if self.model_config:
            model_config.update(self.model_config)
        if self.data_spec:
            model_config.update(
                {
                    "input_channels": self.data_spec["input_channels"],
                    "num_output_classes": [
                        s["num_classes"] for s in self.data_spec["output_spec"]
                    ],
                }
            )
        # create model accordingly
        model_class = getattr(models, self.model_class)
        return model_class(**model_config)

    base_config: str = t.Str()
    model_config: dict = t.Dict()
    model_class: str = t.Enum("FilterNet", "DeepConvLSTM")

    lr_exp: float = t.Float(-3.0)
    batch_size: int = t.Int()
    win_len: int = t.Int(512)
    n_samples_per_batch: int = t.Int(5000)
    train_step: int = t.Int(16)
    seed: int = t.Int()
    decimation: int = t.Int(1)
    optim_type: str = t.Enum(["Adam", "SGD, RMSprop"])
    loss_func: str = t.Enum(["cross_entropy", "binary_cross_entropy"])
    patience: int = t.Int(10)
    lr_decay: float = t.Float(0.95)
    weight_decay: float = t.Float(1e-4)
    alpha: float = t.Float(0.99)
    momentum: float = t.Float(0.25)
    validation_fold: int = t.Int()
    epoch_size: float = t.Float(2.0)
    y_cols: str = t.Str()
    sensor_subset: str = t.Str()

    has_null_class: bool = t.Bool()

    def _has_null_class_default(self):
        return self.data_spec["output_spec"][0]["classes"][0] in ("", "Null")

    predict_null_class: bool = t.Bool(True)

    _class_weights: torch.Tensor = t.Instance(torch.Tensor)

    def __class_weights_default(self):
        # Not weights for now because didn't seem to increase things significantly and
        #   added yet another hyper-parameter. Using zero didn't seem to work well.
        if False and self.has_null_class and not self.predict_null_class:
            cw = torch.ones(self.model.num_output_classes, device=self.device)
            cw[0] = 0.01
            cw /= cw.sum()
            return cw
        return None

    dataset: str = t.Enum(
        ["opportunity", "smartphone_hapt", "har", "intention_recognition"]
    )
    name: str = t.Str()

    def _name_default(self):
        import time

        modelstr = self.model.__class__.__name__
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return f"{modelstr}_{timestr}"

    model_path: str = t.Str()

    def _model_path_default(self):
        return f"saved_models/{self.name}/"

    data_spec: dict = t.Any()
    epoch_iters: int = t.Int(0)
    train_state: TrainState = t.Instance(TrainState, ())
    cp_iter: int = t.Int()

    cuda: bool = t.Bool(transient=True)

    def _cuda_default(self):
        return torch.cuda.is_available()

    device: str = t.Str(transient=True)

    def _device_default(self):
        return "cuda" if self.cuda else "cpu"

    dl_train: DataLoader = t.Instance(DataLoader, transient=True)

    def _dl_train_default(self):
        return self._get_dl("train")

    dl_val: DataLoader = t.Instance(DataLoader, transient=True)

    def _dl_val_default(self):
        return self._get_dl("val")

    dl_test: DataLoader = t.Instance(DataLoader, transient=True)

    def _dl_test_default(self):
        return self._get_dl("test")

    def _get_dl(self, s):

        if self.dataset == "opportunity":
            from datasets.opportunity import get_x_y_contig
        elif self.dataset == "smartphone_hapt":
            from datasets.smartphone_hapt import get_x_y_contig
        elif self.dataset == "har":
            from datasets.har import get_x_y_contig
        elif self.dataset == "intention_recognition":
            from datasets.intention_recognition import get_x_y_contig
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        kwargs = {}
        if self.y_cols:
            kwargs["y_cols"] = self.y_cols
        if self.sensor_subset:
            kwargs["sensor_subset"] = self.sensor_subset

        Xc, ycs, data_spec = get_x_y_contig(s, **kwargs)

        if s == "train":
            # Training shuffles, and we set epoch size to length of the dataset. We can set train_step as
            # small as we want to get more windows; we'll only run len(Sc)/win_len of them in each training
            # epoch.
            self.epoch_iters = int(len(Xc) / self.decimation)
            X, ys = sliding_window_x_y(
                Xc, ycs, win_len=self.win_len, step=self.train_step, shuffle=False
            )
            # Set the overall data spec using the training set,
            #  and modify later if more info is needed.
            self.data_spec = data_spec
        else:
            # Val and test data are not shuffled.
            # Each point is inferred ~twice b/c step = win_len/2
            X, ys = sliding_window_x_y(
                Xc,
                ycs,
                win_len=self.win_len,
                step=int(self.win_len / 2),
                shuffle=False,  # Cannot be true with windows
            )

        dl = DataLoader(
            TensorDataset(torch.Tensor(X), *[torch.Tensor(y).long() for y in ys]),
            batch_size=self.batch_size,
            shuffle=True if s == "train" else False,
        )
        return dl

    def _batch_size_default(self):
        batch_size = int(self.n_samples_per_batch / self.win_len)
        print(f"Batch size: {batch_size}")
        return batch_size

    optimizer = t.Any(transient=True)

    def _optimizer_default(self):
        if self.optim_type == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=10 ** (self.lr_exp),
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optim_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=10 ** (self.lr_exp),
                weight_decay=self.weight_decay,
                amsgrad=True,
            )
        elif self.optim_type == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=10 ** (self.lr_exp),
                alpha=self.alpha,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        else:
            raise NotImplementedError(self.optim_type)
        return optimizer

    iteration: int = t.Property(t.Int)

    def _get_iteration(self):
        return len(self.train_state.epoch_records) + 1

    lr_scheduler = t.Any(transient=True)

    def _lr_scheduler_default(self):
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.lr_decay  # , last_epoch=self._iteration
        )

        # If this is being re-instantiated in mid-training, then we must
        #  iterate scheduler forward to match the training step.
        for i in range(self.iteration):
            if self.lr_decay != 1:
                lr_scheduler.step()

        return lr_scheduler

    #####
    # Training Methods
    ##
    def _train_batch(self, data, targets):
        self.optimizer.zero_grad()
        loss, output, _targets, _ = self._run_model_on_batch(data, targets)
        loss.backward()
        self.optimizer.step()
        # if self.max_lr:
        #     self.lr_scheduler.step()

        return loss, output, _targets

    def _run_model_on_batch(self, data, targets):
        targets = torch.stack(targets)

        if self.cuda:
            data, targets = data.cuda(), targets.cuda()

        output = self.model(data)

        _targets = self.model.transform_targets(targets, one_hot=False)
        if self.loss_func == "cross_entropy":
            _losses = [
                F.cross_entropy(o, t, weight=self._class_weights)
                for o, t in zip(output, _targets)
            ]
            loss = sum(_losses)
        elif self.loss_func == "binary_cross_entropy":
            _targets_onehot = self.model.transform_targets(targets, one_hot=True)
            _losses = [
                F.binary_cross_entropy_with_logits(o, t, weight=self._class_weights)
                for o, t in zip(output, _targets_onehot)
            ]
            loss = sum(_losses)
        else:
            raise NotImplementedError(self.loss)

        # Assume only 1 output:

        return loss, output[0], _targets[0], _losses[0]

    def _calc_validation_loss(self):
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, *targets) in enumerate(self.dl_val):
                loss, _, _, _ = self._run_model_on_batch(data, targets)
                running_loss += loss.item() * data.size(0)

        return running_loss / len(self.dl_val.dataset)

    def _train_epoch(self):

        self.model.train()

        train_losses = []
        train_accs = []

        for batch_idx, (data, *targets) in enumerate(self.dl_train):
            if (
                batch_idx * data.shape[0] * data.shape[2]
                > self.epoch_iters * self.epoch_size
            ):
                # we've effectively finished one epoch worth of data; break!
                break

            batch_loss, batch_output, batch_targets = self._train_batch(data, targets)
            train_losses.append(batch_loss.detach().cpu().item())
            batch_preds = torch.argmax(batch_output, 1, False)
            train_accs.append(
                (batch_preds == batch_targets).detach().cpu().float().mean().item()
            )

        if self.lr_decay != 1:
            self.lr_scheduler.step()

        return EpochMetrics(loss=np.mean(train_losses), acc=np.mean(train_accs))

    def _val_epoch(self):
        return self._eval_epoch(self.dl_val)

    def _eval_epoch(self, data_loader):
        # Validation
        self.model.eval()

        losses = []
        outputs = []
        targets = []

        with torch.no_grad():
            for batch_idx, (data, *target) in enumerate(data_loader):
                (
                    batch_loss,
                    batch_output,
                    batch_targets,
                    train_losses,
                ) = self._run_model_on_batch(data, target)

                losses.append(batch_loss.detach().cpu().item())
                outputs.append(
                    torch.argmax(batch_output, 1, False)
                    .detach()
                    .cpu()
                    .data.numpy()
                    .flatten()
                )
                targets.append(batch_targets.detach().cpu().data.numpy().flatten())

        targets = np.hstack(targets)
        outputs = np.hstack(outputs)
        acc = sklearn.metrics.accuracy_score(targets, outputs)
        f1 = sklearn.metrics.f1_score(targets, outputs, average="weighted")

        return EpochMetrics(loss=np.mean(losses), acc=acc, f1=f1)

    def init_data(self):
        # Initiate loading of datasets, model
        _, _, _ = self.dl_train, self.dl_val, self.dl_test
        _ = self.model

    def init_train(self):

        # initialization
        if self.seed:
            torch.manual_seed(self.seed)
        if self.cuda:
            if self.seed:
                torch.cuda.manual_seed(self.seed)
        self.model.to(self.device)

    def train_one_epoch(self, verbose=True) -> EpochRecord:
        """ traing a single epoch -- method tailored to the Ray.tune methodology."""
        epoch_record = EpochRecord(epoch=len(self.train_state.epoch_records))
        self.train_state.epoch_records.append(epoch_record)

        with Timer("Train Epoch", log_output=verbose) as t:
            epoch_record.train = self._train_epoch()
        epoch_record.iter_s_cpu = t.interval_cpu
        epoch_record.iter_s_wall = t.interval_wall
        epoch_record.lr = self.optimizer.param_groups[0]["lr"]

        with Timer("Val Epoch", log_output=verbose):
            epoch_record.val = self._val_epoch()

        df = self.train_state.to_df()

        # Early stopping / checkpointing implementation
        df["raw_metric"] = df.val_loss / df.val_f1
        df["ewma_smoothed_loss"] = (
            df["raw_metric"].ewm(ignore_na=False, halflife=3).mean()
        )
        df["instability_penalty"] = (
            df["raw_metric"].rolling(5, min_periods=3).std().fillna(0.75)
        )
        stopping_metric = df["stopping_metric"] = (
            df["ewma_smoothed_loss"] + df["instability_penalty"]
        )
        epoch_record.stopping_metric = df["stopping_metric"].iloc[-1]

        idx_this_iter = stopping_metric.index.max()
        idx_best_yet = stopping_metric.idxmin()
        self.train_state.best_sm = df.loc[idx_best_yet, "stopping_metric"]
        self.train_state.best_loss = df.loc[idx_best_yet, "val_loss"]
        self.train_state.best_f1 = df.loc[idx_best_yet, "val_f1"]

        if idx_best_yet == idx_this_iter:
            # Best yet! Checkpoint.
            epoch_record.should_checkpoint = True
            self.cp_iter = epoch_record.epoch

        else:
            if self.patience is not None:
                patience_counter = idx_this_iter - idx_best_yet
                assert patience_counter >= 0
                if patience_counter > self.patience:
                    if verbose:
                        print(
                            f"Early stop! Out of patience ( {patience_counter} > {self.patience} )"
                        )
                    epoch_record.done = True

        if verbose:
            self.print_train_summary()

        return epoch_record

    def train(self, max_epochs=50, verbose=True):
        """ A pretty standard training loop, constrained to stop in `max_epochs` but may stop early if our
        custom stopping metric does not improve for `self.patience` epochs. Always checkpoints
        when a new best stopping_metric is achieved. An alternative to using
        ray.tune for training."""

        self.init_data()
        self.init_train()

        while True:
            epoch_record = self.train_one_epoch(verbose=verbose)

            if epoch_record.should_checkpoint:
                last_cp = self._save()
                if verbose:
                    print(f"<<<< Checkpointed ({last_cp}) >>>")
            if epoch_record.done:
                break
            if epoch_record.epoch >= max_epochs:
                break

        # Save trainer state, but not model"
        self._save(save_model=False)
        if verbose:
            print(self.model_path)

    def print_train_summary(self):
        df = self.train_state.to_df()

        with pd.option_context(
            "display.max_rows",
            100,
            "display.max_columns",
            100,
            "display.precision",
            3,
            "display.width",
            180,
        ):
            print(df.drop(["done"], axis=1, errors="ignore"))

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

        # Load model (after loading state in case we need to re-initialize model from config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Be careful to reinitialize optimizer and lr scheduler
        self.optimizer = self._optimizer_default()
        self.lr_scheduler = self._lr_scheduler_default()


#
# class EnsembleCNNLSTMTrainable(CNNLSTMTrainable):
#     def _setup(self, config={}):
#         """Decimation is for speedup during unit testing only."""
#         super()._setup(config=config)
#         self.model = mo.FilterNetEnsemble(config=config)
