# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base_layers import CGLLayer
from .base_net import BaseNet

DEFAULT_WIDTH = 100


class FilterNet(BaseNet):
    def build(
        self,
        n_pre=1,
        w_pre=DEFAULT_WIDTH,
        n_strided=3,
        w_strided=DEFAULT_WIDTH,
        n_interp=4,
        w_interp=DEFAULT_WIDTH,
        n_dense_pre_l=1,
        w_dense_pre_l=DEFAULT_WIDTH,
        n_l=1,
        w_l=DEFAULT_WIDTH,
        n_dense_post_l=0,
        w_dense_post_l=int(DEFAULT_WIDTH / 2),
        cnn_kernel_size=5,
        scale=1.0,
        bn_pre=False,
        dropout=0.1,
        do_pool=True,
        stride_pos="post",
        stride_amt=2,
        **other_kwargs,
    ):
        # if scale != 1:
        w_pre = int((w_pre * scale))  # / 6) * 6
        w_strided = int((w_strided * scale))  # / 6) * 6
        w_interp = int(w_interp * scale)
        w_dense_pre_l = int(w_dense_pre_l * scale)
        w_l = int((w_l * scale) / 2) * 2
        w_dense_post_l = int(w_dense_post_l * scale)

        down_stack_1 = []
        in_shape = self.input_channels

        if bn_pre:
            down_stack_1.append(nn.BatchNorm1d(in_shape))

        for i in range(n_pre):
            down_stack_1.append(
                CGLLayer(in_shape, w_pre, cnn_kernel_size, type="cnn", dropout=dropout)
            )
            in_shape = down_stack_1[-1].output_size

        for i in range(n_strided):
            stride = stride_amt
            pool = stride if (do_pool and stride > 1) else None
            ltype = "cnn"
            down_stack_1.append(
                CGLLayer(
                    in_shape,
                    w_strided,
                    cnn_kernel_size,
                    type=ltype,
                    stride=stride,
                    pool=pool,
                    stride_pos=stride_pos,
                    dropout=dropout,
                    # groups=3 if ( i % 2 == 0 ) else 2
                )
            )
            self.output_stride *= stride
            in_shape = down_stack_1[-1].output_size
        ds_1_end_size = in_shape
        self.down_stack_1 = nn.Sequential(*down_stack_1)

        ds2_ltype = "cnn"
        down_stack_2 = []

        for i in range(n_interp):
            stride = stride_amt if (i < n_interp - 1) else 1
            pool = stride if (do_pool and stride > 1) else None
            w = int(np.ceil(w_interp * 0.5 ** (i + 1)))
            # if i == n_interp-1:
            #     w = int(w_interp * .66)
            # if i == n_interp - 2:
            #     w =int(w_interp * .33)
            # else:
            #     w = w_interp
            down_stack_2.append(
                CGLLayer(
                    in_shape,
                    w,
                    cnn_kernel_size,
                    type=ds2_ltype,
                    stride=stride,
                    pool=pool,
                    stride_pos=stride_pos,
                    dropout=dropout,
                    # groups = 3 if ( i % 2 == 0 ) else 2
                )
            )
            in_shape = down_stack_2[-1].output_size

        self.down_stack_2 = nn.Sequential(*down_stack_2)

        self.merged_output_size = ds_1_end_size + sum(
            [l.output_size for l in down_stack_2]
        )

        in_shape = self.merged_output_size

        lstm_stack = []
        for i in range(n_dense_pre_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape, w_dense_pre_l, kernel_size=1, type="cnn", dropout=dropout
                )
            )
            in_shape = lstm_stack[-1].output_size

        for i in range(n_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape,
                    w_l,
                    cnn_kernel_size,  # unused when type!-='cnn'
                    type="lstm",
                    dropout=dropout,
                )
            )
            in_shape = lstm_stack[-1].output_size

        for i in range(n_dense_post_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape, w_dense_post_l, kernel_size=1, type="cnn", dropout=dropout
                )
            )
            in_shape = lstm_stack[-1].output_size

        self.lstm_stack = nn.Sequential(*lstm_stack)

        # [batch, chan, seq]

        end_stacks = []
        for num_output_classes in self.num_output_classes:
            end_stacks.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    #     # This sort of Conv1D acts as a time-distributed Dense layer.
                    nn.Linear(in_shape, num_output_classes),
                    # nn.Conv1d(
                    #     in_shape, num_output_classes, 1
                    # ),  # time-distributed dense
                )
                # CGLLayer(
                #     in_shape,
                #     num_output_classes,
                #     kernel_size=1,
                #     type="cnn",
                #     dropout=dropout,
                #     batch_norm=False
                # )
            )

        self.end_stacks = nn.ModuleList(end_stacks)

    def _forward(self, X, **kwargs):
        """(N, C_{in}, L_{in})"""
        Xs = [X]  # [batch, chan, seq]
        Xs.append(self.down_stack_1(Xs[-1]))

        to_merge = [Xs[-1]]
        for module in self.down_stack_2:
            output = module(Xs[-1])
            Xs.append(output)
            to_merge.append(
                F.interpolate(
                    output,
                    size=to_merge[0].shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            )

        merged = torch.cat(to_merge, dim=1)
        Xs.append(merged)
        Xs.append(self.lstm_stack(Xs[-1]))

        if self.keep_intermediates:
            self.Xs = Xs

        ys = []

        # (N, C_{in}, L_{in})

        for end_stack in self.end_stacks:
            # (N, C_{in}, L_{in}) => # (N, L_{in},  C_{in},)
            x = Xs[-1].permute([0, 2, 1])
            x = end_stack(x)
            x = x.permute([0, 2, 1])
            ys.append(x)

        # ys = [end_stack(Xs[-1]) for end_stack in self.end_stacks]

        # No softmax because the pytorch cross_entropy loss function wants the raw outputs.

        return ys
