# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import numpy as np
from torch import nn


class CustomRNNMixin(object):
    """Mixin to hekp wraps PyTorch recurrent layer(s) to swap axes 1&2 (and back) since that's what PyTorch RNNs expect.
    """

    def __init__(self, *args, **kwargs):
        if "batch_first" not in kwargs:
            kwargs["batch_first"] = True
        super().__init__(*args, **kwargs)

    def forward(self, input):
        input = input.transpose(1, 2).contiguous()
        output, h_n = super().forward(input)
        return output.transpose(1, 2).contiguous()


class CustomGRU(CustomRNNMixin, nn.GRU):
    """Wraps PyTorch GRU to swap axes 1&2 (and back) since that's what PyTorch RNNs expect.
    GRU sub-type.
    """

    pass


class CustomLSTM(CustomRNNMixin, nn.LSTM):
    """Wraps PyTorch LSTM version to swap axes 1&2 (and back) since that's what PyTorch RNNs expect.
    LSTM sub-type.
    """

    pass


class CGLLayer(nn.Sequential):
    """Flexible mplementation of a convolution/GRU/LSTM layer, which is the basic building block of our models. Each
    layer is made up of (optional) dropout, a CNN, GRU, or LSTM layer surrounded by (optional) striding/pooling
    layers, and a BatchNorm layer.

    This layer subclasses torch.nn.Sequential so that all the pytorch magic still works with it (like transferring
    to/from devices, initializing weights, switching back/forth to eval mode, etc)
    """

    output_size = (
        None
    )  # type: int # depth (channels) output by this layer, useful for hooking up to subsequent modules.

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=5,
        type="cnn",
        stride=1,
        pool=None,
        dropout=0.1,
        stride_pos=None,
        batch_norm=True,
        groups=1,
    ):
        """

        Parameters
        ----------
        input_size: int
            Depth (channels) of input / previous layer
        output_size: int
            Depth (channels) that this layer will output
        kernel_size: int
            For CNNs
        type: str
            'cnn', 'lstm', or 'gru'; determines primary layer type.
        stride: int
            How much to decimate output (in temporal dimension) via _striding_. Defaults to 1 (no decimation).
        pool: int
            How much to decimate output (in temporal dimension) via _average_pooling_. Defaults to 1 (no decimation).
        dropout: float
            Amount of dropout Defaults to 0.0, i.e., None
        stride_pos: str
            For recurrent layers only, determines whether striding/pooling is done *before* (default) or
            *after* the recurrent layer.
        batch_norm: bool
            If True (default), the activation layer is followed by a batchnorm layer.
        """

        layers = []
        self.output_size = output_size

        if type == "cnn":
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            s = 1 if pool else stride
            p = int(np.ceil((kernel_size - s) / 2.0))
            layers.append(
                nn.Conv1d(
                    input_size,
                    output_size,
                    stride=s,
                    kernel_size=kernel_size,
                    padding=p,
                    groups=groups,
                )
            )
            layers.append(nn.ReLU())
            if pool:
                p = int(np.ceil((pool - stride) / 2.0))
                layers.append(
                    nn.AvgPool1d(pool, stride, padding=p, count_include_pad=False)
                )
        elif type in ["gru", "lstm"]:
            klass = {"gru": CustomGRU, "lstm": CustomLSTM}[type]
            if (pool or stride) and stride_pos != "post":
                pl = 1 if not pool else pool
                p = np.ceil((pl - stride) / 2.0).astype(int)
                layers.append(nn.AvgPool1d(pl, stride=stride, padding=p))
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            assert output_size % 2 == 0  # must be even b/c bidirectional
            layers.append(
                klass(
                    input_size=input_size,
                    hidden_size=int(output_size / 2),
                    bidirectional=True,
                )
            )
            if (pool or stride) and stride_pos == "post":
                pl = 1 if not pool else pool
                p = np.ceil((pl - stride) / 2.0).astype(int)
                layers.append(nn.AvgPool1d(pl, stride=stride, padding=p))
        else:
            raise ValueError("Unknown layer type: %s" % type)

        # Follow with BN
        if batch_norm:
            layers.append(nn.BatchNorm1d(self.output_size))

        super().__init__(*layers)
