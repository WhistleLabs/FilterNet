# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import numpy as np
import torch
from torch import nn


class BaseNet(nn.Module):
    """ Abstract 'base' network that can be reimplemented for specific architectures."""

    def __init__(
        self,
        input_channels=113,
        num_output_classes=[18],
        output_type="many_to_many",
        keep_intermediates=False,
        **other_kwargs,
    ):

        self.output_type = output_type
        self.num_output_classes = num_output_classes
        self.input_channels = input_channels
        self.keep_intermediates = keep_intermediates
        self.padding_lost_per_side = 0
        self.output_stride = 1

        super(BaseNet, self).__init__()

        self.build(**other_kwargs)

    def build(self, **other_kwargs):
        """ Builds the network. Can take any number of custom params as kwargs to configure it.
        REIMPLEMENT IN SUBCLASSES.
        """
        raise NotImplementedError()

    def forward(self, X, **kwargs):
        ys = self._forward(X, **kwargs)

        if self.output_type == "many_to_one_takelast":
            return [y[:, :, [-1]] for y in ys]
        elif self.output_type == "many_to_many":
            return ys
        else:
            raise NotImplemented(self.output_type)

    def _forward(self, X, **kwargs):
        """Forward pass logic specific to this network type.
        REIPMLEMENT IN SUBCLASSES.
        Input dimensionality: (N, C_{in}, L_{in})"""
        raise NotImplementedError()

    def transform_targets(self, ys, one_hot=True):
        """ Convert a `y` vector (one of `ys`) into targets that can be compared
        to network outputs... take into account padding, one-hot encoding (if requested),
        and whether the network is many-to-many or many-to-one. """
        ys2 = []
        for i_y, y in enumerate(ys):
            if self.output_type == "many_to_one_takelast" and not one_hot:
                ys2.append(y[:, [-1]])
                continue

            # Account for any change in sequence length due to padding
            if self.padding_lost_per_side > 0:
                y = y[:, self.padding_lost_per_side : -self.padding_lost_per_side]

            # for many-to-many, if needed:
            win_len = y.shape[-1]
            # Calculate number of outputs. This is not always accurate and sometimes
            # 'floor' needs to change to 'ceil' or vice-versa... TBD is to implement
            # a system that calculates this accurately for all of our possible
            # architectures.
            output_size = int(np.floor(win_len / float(self.output_stride)))
            # Now, create that many outputs, evenly spaced by output_stride
            output_idxs = np.arange(output_size) * self.output_stride
            # Now, center it in the middle of the window. Depending on our
            #  architecture, this many not be *exactly* optimal, but it's
            #  a good guess on average.
            # Note: win_len - 1 because of zero-indexing
            output_idxs = np.round(
                output_idxs - (output_idxs.mean() - (win_len - 1) / 2.0)
            ).astype(int)

            if one_hot:
                if len(y.shape) == 2:
                    # Do one-hot encoding
                    y = torch.zeros(
                        y.size()[0],
                        self.num_output_classes[i_y],
                        y.size()[1],
                        device=y.device,
                    ).scatter_(1, y.unsqueeze(1), 1)

                if self.output_type == "many_to_one_takelast":
                    ys2.append(y[:, :, [output_idxs[-1]]])
                elif self.output_type == "many_to_many":
                    ys2.append(y[:, :, output_idxs])
                else:
                    raise NotImplemented(self.output_type)

            else:
                if self.output_type == "many_to_one_takelast":
                    ys2.append(y[:, [output_idxs[-1]]])
                elif self.output_type == "many_to_many":
                    ys2.append(y[:, output_idxs])
                else:
                    raise NotImplemented(self.output_type)

        return ys2
