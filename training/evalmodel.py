# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import os
import pickle
import typing as ty
from builtins import AssertionError

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.optim
import traits.api as t
from scipy.special import softmax
from torch.utils.data import DataLoader

import models as mo
from mputil import Timer
from training.train import Trainer


class ClassWardMetrics(t.HasStrictTraits):
    segment_twoset_results: dict = t.Dict()
    event_detailed_scores: dict = t.Dict()
    event_standard_scores: dict = t.Dict()


class WardMetrics(t.HasStrictTraits):
    class_ward_metrics: ty.List[ClassWardMetrics] = t.List(ClassWardMetrics, [])
    overall_ward_metrics: ClassWardMetrics = t.Instance(ClassWardMetrics)
    df_event_scores: pd.DataFrame = t.Instance(pd.DataFrame())
    df_event_detailed_scores: pd.DataFrame = t.Instance(pd.DataFrame())
    df_segment_2set_results: pd.DataFrame = t.Instance(pd.DataFrame())


class EvalModel(t.HasStrictTraits):
    trainer: Trainer = t.Any()
    model: mo.BaseNet = t.DelegatesTo("trainer")
    dl_test: DataLoader = t.DelegatesTo("trainer")
    data_spec: dict = t.DelegatesTo("trainer")
    cuda: bool = t.DelegatesTo("trainer")
    device: str = t.DelegatesTo("trainer")
    loss_func: str = t.DelegatesTo("trainer")
    model_path: str = t.DelegatesTo("trainer")
    has_null_class: bool = t.DelegatesTo("trainer")
    predict_null_class: bool = t.DelegatesTo("trainer")

    # 'prediction' mode employs overlap and reconstructs signal
    #   as a contiguous timeseries w/ optional windowing.
    #   It aims for best accuracy/f1 by using overlap, and will
    #   typically outperform 'training' mode.
    # 'training' mode does not average repeated point and does
    #   not window; it should product acc/loss/f1 similar to
    #   training mode.
    run_mode: str = t.Enum(["prediction", "training"])
    window: str = t.Enum(["hanning", "boxcar"])
    eval_batch_size: int = t.Int(100)

    target_names: ty.List[str] = t.ListStr()

    def _target_names_default(self):
        target_names = self.data_spec["output_spec"][0]["classes"]

        if self.has_null_class:
            assert target_names[0] in ("", "Null")

            if not self.predict_null_class:
                target_names = target_names[1:]

        return target_names

    def _run_model_on_batch(self, data, targets):
        targets = torch.stack(targets)

        if self.cuda:
            data, targets = data.cuda(), targets.cuda()

        output = self.model(data)

        _targets = self.model.transform_targets(targets, one_hot=False)
        if self.loss_func == "cross_entropy":
            _losses = [F.cross_entropy(o, t) for o, t in zip(output, _targets)]
            loss = sum(_losses)
        elif self.loss_func == "binary_cross_entropy":
            _targets_onehot = self.model.transform_targets(targets, one_hot=True)
            _losses = [
                F.binary_cross_entropy_with_logits(o, t)
                for o, t in zip(output, _targets_onehot)
            ]
            loss = sum(_losses)
        else:
            raise NotImplementedError(self.loss)

        # Assume only 1 output:

        return loss, output[0], _targets[0], _losses[0]

    def run_test_set(self, dl=None):
        """ Runs `self.model` on `self.dl_test` (or a provided dl) and stores results for subsequent evaluation. """
        if dl is None:
            dl = self.dl_test

        if self.cuda:
            self.model.cuda()
        self.model.eval()
        if self.eval_batch_size:
            dl = DataLoader(dl.dataset, batch_size=self.eval_batch_size, shuffle=False)
        #
        #     # Xc, yc = data.get_x_y_contig('test')
        X, *ys = dl.dataset.tensors
        # X: [N, input_chans, win_len]
        step = int(X.shape[2] / 2)
        assert torch.equal(X[0, :, step], X[1, :, 0])

        losses = []
        outputsraw = []
        outputs = []
        targets = []

        with Timer("run", log_output=False) as tr:
            with Timer("infer", log_output=False) as ti:
                for batch_idx, (data, *target) in enumerate(dl):
                    (
                        batch_loss,
                        batch_output,
                        batch_targets,
                        train_losses,
                    ) = self._run_model_on_batch(data, target)

                    losses.append(batch_loss.detach().cpu().item())
                    outputsraw.append(batch_output.detach().cpu().data.numpy())
                    outputs.append(
                        torch.argmax(batch_output, 1, False).detach().cpu().data.numpy()
                    )
                    targets.append(batch_targets.detach().cpu().data.numpy())
            self.infer_time_s_cpu = ti.interval_cpu
            self.infer_time_s_wall = ti.interval_wall

            self.loss = np.mean(losses)
            targets = np.concatenate(targets, axis=0)  # [N, out_win_len]
            outputsraw = np.concatenate(
                outputsraw, axis=0
            )  # [N, n_out_classes, out_win_len]
            outputs = np.concatenate(outputs, axis=0)  # [N, n_out_classes, out_win_len]

            # win_len = toutputsraw[0].shape[-1]
            if (
                self.model.output_type == "many_to_one_takelast"
                or self.run_mode == "training"
            ):
                self.targets = np.concatenate(targets, axis=-1)  # [N,]
                self.outputsraw = np.concatenate(
                    outputsraw, axis=-1
                )  # [n_out_classes, N,]
                self.outputs = np.concatenate(outputs, axis=-1)  # [N,]

            elif self.run_mode == "prediction":
                n_segments, n_classes, out_win_len = outputsraw.shape

                output_step = int(out_win_len / 2)

                if self.window == "hanning":
                    EPS = 0.001  # prevents divide-by-zero
                    arr_window = (1 - EPS) * np.hanning(out_win_len) + EPS
                elif self.window == "boxcar":
                    arr_window = np.ones((out_win_len,))
                else:
                    raise ValueError()

                # Allocate space for merged predictions
                if self.has_null_class and not self.predict_null_class:
                    outputsraw2 = np.zeros(
                        (n_segments + 1, n_classes - 1, output_step, 2)
                    )
                    window2 = np.zeros(
                        (n_segments + 1, n_classes - 1, output_step, 2)
                    )  # [N+1, out_win_len/2, 2]
                    # Drop in outputs/window vals in the two layers
                    outputsraw = outputsraw[:, 1:, :]
                else:
                    outputsraw2 = np.zeros((n_segments + 1, n_classes, output_step, 2))
                    window2 = np.zeros(
                        (n_segments + 1, n_classes, output_step, 2)
                    )  # [N+1, out_win_len/2, 2]

                # Drop in outputs/window vals in the two layers
                outputsraw2[:-1, :, :, 0] = outputsraw[:, :, :output_step]
                outputsraw2[1:, :, :, 1] = outputsraw[
                    :, :, output_step : output_step * 2
                ]
                window2[:-1, :, :, 0] = arr_window[:output_step]
                window2[1:, :, :, 1] = arr_window[output_step : output_step * 2]

                merged_outputsraw = (outputsraw2 * window2).sum(axis=3) / (window2).sum(
                    axis=3
                )
                softmaxed_merged_outputsraw = softmax(merged_outputsraw, axis=1)
                merged_outputs = np.argmax(softmaxed_merged_outputsraw, 1)

                self.outputsraw = np.concatenate(merged_outputsraw, axis=-1)
                self.outputs = np.concatenate(merged_outputs, axis=-1)
                self.targets = np.concatenate(
                    np.concatenate(
                        [
                            targets[:, :output_step],
                            targets[[-1], output_step : output_step * 2],
                        ],
                        axis=0,
                    ),
                    axis=-1,
                )
            else:
                raise ValueError()

        if self.has_null_class and not self.predict_null_class:
            not_null_mask = self.targets > 0
            self.outputsraw = self.outputsraw[..., not_null_mask]
            self.outputs = self.outputs[not_null_mask]
            self.targets = self.targets[not_null_mask]
            self.targets -= 1

        self.n_samples_in = np.prod(dl.dataset.tensors[1].shape)
        self.n_samples_out = len(self.outputs)
        self.infer_samples_per_s = self.n_samples_in / self.infer_time_s_wall
        self.run_time_s_cpu = tr.interval_cpu
        self.run_time_s_wall = tr.interval_wall

    loss: float = t.Float()
    targets: np.ndarray = t.Array()
    outputsraw: np.ndarray = t.Array()
    outputs: np.ndarray = t.Array()
    n_samples_in: int = t.Int()
    n_samples_out: int = t.Int()
    infer_samples_per_s: float = t.Float()

    infer_time_s_cpu: float = t.Float()
    infer_time_s_wall: float = t.Float()
    run_time_s_cpu: float = t.Float()
    run_time_s_wall: float = t.Float()

    extra: dict = t.Dict({})

    acc: float = t.Float()
    f1: float = t.Float()
    f1_mean: float = t.Float()
    event_f1: float = t.Float()
    classification_report_txt: str = t.Str()
    classification_report_dict: dict = t.Dict()
    classification_report_df: pd.DataFrame = t.Property(t.Instance(pd.DataFrame))
    confusion_matrix: np.ndarray = t.Array()

    nonull_acc: float = t.Float()
    nonull_f1: float = t.Float()
    nonull_f1_mean: float = t.Float()
    nonull_classification_report_txt: str = t.Str()
    nonull_classification_report_dict: dict = t.Dict()
    nonull_classification_report_df: pd.DataFrame = t.Property(t.Instance(pd.DataFrame))
    nonull_confusion_matrix: np.ndarray = t.Array()

    def calc_metrics(self):

        self.acc = sklearn.metrics.accuracy_score(self.targets, self.outputs)
        self.f1 = sklearn.metrics.f1_score(
            self.targets, self.outputs, average="weighted"
        )
        self.f1_mean = sklearn.metrics.f1_score(
            self.targets, self.outputs, average="macro"
        )

        self.classification_report_txt = sklearn.metrics.classification_report(
            self.targets,
            self.outputs,
            digits=3,
            labels=np.arange(len(self.target_names)),
            target_names=self.target_names,
        )
        self.classification_report_dict = sklearn.metrics.classification_report(
            self.targets,
            self.outputs,
            digits=3,
            output_dict=True,
            labels=np.arange(len(self.target_names)),
            target_names=self.target_names,
        )
        self.confusion_matrix = sklearn.metrics.confusion_matrix(
            self.targets, self.outputs
        )

        # Now, ignoring the null/none class:
        if self.has_null_class and self.predict_null_class:
            # assume null class comes fistnonull_mask = self.targets > 0
            nonull_mask = self.targets > 0
            nonull_targets = self.targets[nonull_mask]
            # nonull_outputs = self.outputs[nonull_mask]
            nonull_outputs = self.outputsraw[1:, :].argmax(axis=0)[nonull_mask] + 1

            self.nonull_acc = sklearn.metrics.accuracy_score(
                nonull_targets, nonull_outputs
            )
            self.nonull_f1 = sklearn.metrics.f1_score(
                nonull_targets, nonull_outputs, average="weighted"
            )
            self.nonull_f1_mean = sklearn.metrics.f1_score(
                nonull_targets, nonull_outputs, average="macro"
            )
            self.nonull_classification_report_txt = sklearn.metrics.classification_report(
                nonull_targets,
                nonull_outputs,
                digits=3,
                labels=np.arange(len(self.target_names)),
                target_names=self.target_names,
            )
            self.nonull_classification_report_dict = sklearn.metrics.classification_report(
                nonull_targets,
                nonull_outputs,
                digits=3,
                output_dict=True,
                labels=np.arange(len(self.target_names)),
                target_names=self.target_names,
            )
            self.nonull_confusion_matrix = sklearn.metrics.confusion_matrix(
                nonull_targets, nonull_outputs
            )
        else:
            self.nonull_acc = self.acc
            self.nonull_f1 = self.f1
            self.nonull_f1_mean = self.f1_mean
            self.nonull_classification_report_txt = self.classification_report_txt
            self.nonull_classification_report_dict = self.classification_report_dict
            self.nonull_confusion_matrix = self.confusion_matrix

    ward_metrics: WardMetrics = t.Instance(WardMetrics)

    def calc_ward_metrics(self):
        """ Do event-wise metrics, using the `wardmetrics` package which implements metrics from:

         [1]    J. A. Ward, P. Lukowicz, and H. W. Gellersen, “Performance metrics for activity recognition,”
                    ACM Trans. Intell. Syst. Technol., vol. 2, no. 1, pp. 1–23, Jan. 2011.
        """

        import wardmetrics

        # Must be in prediction mode -- otherwise, data is not contiguous, ward metrics will be bogus
        assert self.run_mode == "prediction"

        targets = self.targets
        predictions = self.outputs

        wmetrics = WardMetrics()

        targets_events = wardmetrics.frame_results_to_events(targets)
        preds_events = wardmetrics.frame_results_to_events(predictions)

        for i, class_name in enumerate(self.target_names):
            class_wmetrics = ClassWardMetrics()

            t = targets_events.get(str(i), [])
            p = preds_events.get(str(i), [])
            # class_wmetrics['t'] = t
            # class_wmetrics['p'] = p

            try:
                assert len(t) and len(p)
                (
                    twoset_results,
                    segments_with_scores,
                    segment_counts,
                    normed_segment_counts,
                ) = wardmetrics.eval_segments(t, p)
                class_wmetrics.segment_twoset_results = twoset_results

                (
                    gt_event_scores,
                    det_event_scores,
                    detailed_scores,
                    standard_scores,
                ) = wardmetrics.eval_events(t, p)
                class_wmetrics.event_detailed_scores = detailed_scores
                class_wmetrics.event_standard_scores = standard_scores
            except (AssertionError, ZeroDivisionError) as e:
                class_wmetrics.segment_twoset_results = {}
                class_wmetrics.event_detailed_scores = {}
                class_wmetrics.event_standard_scores = {}
                # print("Empty Results or targets for a class.")
                # raise ValueError("Empty Results or targets for a class.")

            wmetrics.class_ward_metrics.append(class_wmetrics)

        tt = []
        pp = []
        for i, class_name in enumerate(self.target_names):
            # skip null class for combined eventing:
            if class_name in ("", "Null"):
                continue

            if len(tt) or len(pp):
                offset = np.max(tt + pp) + 2
            else:
                offset = 0
            [(a + offset, b + offset) for (a, b) in t]

            t = targets_events.get(str(i), [])
            p = preds_events.get(str(i), [])

            tt += [(a + offset, b + offset) for (a, b) in t]
            pp += [(a + offset, b + offset) for (a, b) in p]

        t = tt
        p = pp

        class_wmetrics = ClassWardMetrics()
        assert len(t) and len(p)
        (
            twoset_results,
            segments_with_scores,
            segment_counts,
            normed_segment_counts,
        ) = wardmetrics.eval_segments(t, p)
        class_wmetrics.segment_twoset_results = twoset_results

        (
            gt_event_scores,
            det_event_scores,
            detailed_scores,
            standard_scores,
        ) = wardmetrics.eval_events(t, p)
        class_wmetrics.event_detailed_scores = detailed_scores
        class_wmetrics.event_standard_scores = standard_scores

        # Reformat as dataframe for easier calculations
        df = pd.DataFrame(
            [cm.event_standard_scores for cm in wmetrics.class_ward_metrics],
            index=self.target_names,
        )
        df.loc["all_nonull"] = class_wmetrics.event_standard_scores

        # Calculate F1's to summarize recall/precision for each class
        df["f1"] = (
            2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])
        )
        df["f1 (weighted)"] = (
            2
            * (df["precision (weighted)"] * df["recall (weighted)"])
            / (df["precision (weighted)"] + df["recall (weighted)"])
        )

        # Load dataframes into dictionary output
        wmetrics.df_event_scores = df
        wmetrics.df_event_detailed_scores = pd.DataFrame(
            [cm.event_detailed_scores for cm in wmetrics.class_ward_metrics],
            index=self.target_names,
        )
        wmetrics.df_segment_2set_results = pd.DataFrame(
            [cm.segment_twoset_results for cm in wmetrics.class_ward_metrics],
            index=self.target_names,
        )
        wmetrics.overall_ward_metrics = class_wmetrics

        self.ward_metrics = wmetrics
        self.event_f1 = self.ward_metrics.df_event_scores.loc["all_nonull", "f1"]

    def _get_classification_report_df(self):
        df = pd.DataFrame(self.classification_report_dict).T

        # Include Ward-metrics-derived "Event F1 (unweighted by length)"
        if self.ward_metrics:
            df["event_f1"] = self.ward_metrics.df_event_scores["f1"]
        else:
            df["event_f1"] = np.nan

            # Calculate various summary averages
        df.loc["macro avg", "event_f1"] = df["event_f1"].iloc[:-3].mean()
        df.loc["weighted avg", "event_f1"] = (
            df["event_f1"].iloc[:-3] * df["support"].iloc[:-3]
        ).sum() / df["support"].iloc[:-3].sum()

        df["support"] = df["support"].astype(int)

        return df

    def _get_nonull_classification_report_df(self):
        target_names = self.target_names
        if not (target_names[0] in ("", "Null")):
            return None

        df = pd.DataFrame(self.nonull_classification_report_dict).T

        df["support"] = df["support"].astype(int)

        return df

    def _save(self, checkpoint_dir=None):
        """ Saves/checkpoints model state and training state to disk. """
        if checkpoint_dir is None:
            checkpoint_dir = self.model_path

        os.makedirs(checkpoint_dir, exist_ok=True)

        # save model params
        evalmodel_path = os.path.join(checkpoint_dir, "evalmodel.pth")

        with open(evalmodel_path, "wb") as f:
            pickle.dump(self, f)

        return checkpoint_dir

    def _restore(self, checkpoint_dir=None):
        """ Restores model state and training state from disk. """

        if checkpoint_dir is None:
            checkpoint_dir = self.model_path

        evalmodel_path = os.path.join(checkpoint_dir, "evalmodel.pth")

        # Reconstitute old trainer and copy state to this trainer.
        with open(evalmodel_path, "rb") as f:
            other_evalmodel = pickle.load(f)

        self.__setstate__(other_evalmodel.__getstate__())

        self.trainer._restore(checkpoint_dir)


def load_eval_model_from_dir(checkpoint_dir: str):
    em = EvalModel()
    em._restore(checkpoint_dir)
    return em
