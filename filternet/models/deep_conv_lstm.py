# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

from torch import nn

from .base_net import BaseNet


class DeepConvLSTM(BaseNet):
    """ A pytorch implementation of 'DeepConvLSTM' as described in:

      [1]   F. J. Ordóñez and D. Roggen, “Deep Convolutional and LSTM Recurrent Neural Networks for
            Multimodal Wearable Activity Recognition,” Sensors, vol. 16, no. 1, p. 115, Jan. 2016.
    """

    def __init__(self, **other_kwargs):
        super().__init__(output_type="many_to_one_takelast", **other_kwargs)

    def build(
        self,
        num_filters=64,
        filter_size=5,
        num_units_lstm=128,
        scale=1.0,
        **other_kwargs,
    ):

        pad = 0

        num_filters = int(num_filters * scale)
        num_units_lstm = int(num_units_lstm * scale)

        n_conv = 4
        conv_stack = []
        in_shape = 1
        for i in range(n_conv):
            conv_stack.append(
                nn.Conv2d(in_shape, num_filters, (filter_size, 1), padding=(pad, 0))
            )
            conv_stack.append(nn.ReLU())
            self.padding_lost_per_side += int((filter_size - 1) / 2)
            in_shape = num_filters

        self.conv_stack = nn.Sequential(*conv_stack)

        self.drop1 = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(num_filters * self.input_channels, num_units_lstm)
        self.drop2 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(num_units_lstm, num_units_lstm)

        end_stacks = []
        for num_output_classes in self.num_output_classes:
            end_stacks.append(
                nn.Sequential(
                    nn.Dropout(0.5),
                    # This sort of Conv1D acts as a time-distributed Dense layer.
                    nn.Conv1d(
                        num_units_lstm, num_output_classes, 1
                    ),  # time-distributed dense
                )
            )
        self.end_stacks = nn.ModuleList(end_stacks)

        # Original DeepConvLSTM used an orthogonal weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.orthogonal_(m.weight)

    def _forward(self, X, **kwargs):
        """(N, C_{in}, L_{in})"""
        Xs = [X]  # [batch, chan, seq]
        Xs.append(
            Xs[-1].unsqueeze(1).permute([0, 1, 3, 2])
        )  # [batch, filters, seq, sensors]

        Xs.append(self.conv_stack(Xs[-1]))

        Xs.append(
            Xs[-1].permute([2, 0, 1, 3]).flatten(2)
        )  # to [seq, batch, (filtersxsensors)]

        Xs.append(self.drop1(Xs[-1]))
        Xs.append(self.lstm1(Xs[-1])[0])

        Xs.append(self.drop2(Xs[-1]))
        Xs.append(self.lstm2(Xs[-1])[0])

        Xs.append(Xs[-1].permute([1, 2, 0]))  # back to [batch, chan, seq]

        ys = [end_stack(Xs[-1]) for end_stack in self.end_stacks]
        # No softmax because the pytorch cross_entropy loss function wants the raw outputs.

        if self.keep_intermediates:
            self.Xs = Xs

        return ys
