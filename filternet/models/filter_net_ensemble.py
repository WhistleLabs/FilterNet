# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import torch
from torch import nn

from .base_net import BaseNet


class FilterNetEnsemble(BaseNet):
    variance_penalty = 0.0

    def build(self, **config):
        pass

    def set_models(self, models):
        self.model = nn.ModuleList([m for m in models])

    def _forward(self, X, **kwargs):
        """(N, C_{in}, L_{in})"""
        outputs_list = [sub_model(X) for sub_model in self.model]
        outputs = []
        for i in range(len(self.model[0].num_output_classes)):
            output_ = torch.stack([_outputs[i] for _outputs in outputs_list])
            if self.variance_penalty:
                s = torch.std(output_, dim=0)
                mean = output = torch.mean(output_, dim=0)
                output = mean - self.variance_penalty * s
            else:
                output = torch.mean(output_, dim=0)
            outputs.append(output)
            # output, _ = torch.median(outputs, dim=0)
            # output = torch.log(torch.mean(torch.softmax(outputs, dim=2), dim=0))

        return outputs

    def transform_targets(self, y, one_hot=True):
        return self.model[0].transform_targets(y, one_hot=one_hot)
