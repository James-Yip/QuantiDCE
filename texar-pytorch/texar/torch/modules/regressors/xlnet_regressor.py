# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
XLNet Regressors.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from texar.torch.core.layers import get_initializer
from texar.torch.hyperparams import HParams
from texar.torch.modules.encoders.xlnet_encoder import XLNetEncoder
from texar.torch.modules.regressors.regressor_base import RegressorBase
from texar.torch.modules.pretrained.xlnet import PretrainedXLNetMixin
from texar.torch.modules.pretrained.xlnet_utils import (
    init_weights, params_except_in)
from texar.torch.utils.utils import dict_fetch


__all__ = [
    "XLNetRegressor",
]


class XLNetRegressor(RegressorBase, PretrainedXLNetMixin):
    r"""Regressor based on XLNet modules. Please see
    :class:`~texar.torch.modules.PretrainedXLNetMixin` for a brief description
    of XLNet.

    Arguments are the same as in
    :class:`~texar.torch.modules.XLNetEncoder`.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``xlnet-based-cased``). Please refer to
            :class:`~texar.torch.modules.PretrainedXLNetMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):

        super().__init__(hparams=hparams)

        # Create the underlying encoder
        encoder_hparams = dict_fetch(hparams, XLNetEncoder.default_hparams())

        self._encoder = XLNetEncoder(
            pretrained_model_name=pretrained_model_name,
            cache_dir=cache_dir,
            hparams=encoder_hparams)

        # TODO: The logic here is very similar to that in XLNetClassifier.
        #  We need to reduce the code redundancy.
        if self._hparams.use_projection:
            if self._hparams.regr_strategy == 'all_time':
                self.projection = nn.Linear(
                    self._encoder.output_size * self._hparams.max_seq_length,
                    self._encoder.output_size * self._hparams.max_seq_length)
            else:
                self.projection = nn.Linear(self._encoder.output_size,
                                            self._encoder.output_size)
        self.dropout = nn.Dropout(self._hparams.dropout)

        logit_kwargs = self._hparams.logit_layer_kwargs
        if logit_kwargs is None:
            logit_kwargs = {}
        elif not isinstance(logit_kwargs, HParams):
            raise ValueError("hparams['logit_layer_kwargs'] "
                             "must be a dict.")
        else:
            logit_kwargs = logit_kwargs.todict()

        if self._hparams.regr_strategy == 'all_time':
            self.hidden_to_logits = nn.Linear(
                self._encoder.output_size * self._hparams.max_seq_length,
                1, **logit_kwargs)
        else:
            self.hidden_to_logits = nn.Linear(
                self._encoder.output_size, 1, **logit_kwargs)

        if self._hparams.initializer:
            initialize = get_initializer(self._hparams.initializer)
            assert initialize is not None
            if self._hparams.use_projection:
                initialize(self.projection.weight)
                initialize(self.projection.bias)
            initialize(self.hidden_to_logits.weight)
            if self.hidden_to_logits.bias:
                initialize(self.hidden_to_logits.bias)
        else:
            if self._hparams.use_projection:
                self.projection.apply(init_weights)
            self.hidden_to_logits.apply(init_weights)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in XLNetEncoder
                ...
                # (2) Additional hyperparameters
                "regr_strategy": "cls_time",
                "use_projection": True,
                "logit_layer_kwargs": None,
                "name": "xlnet_regressor",
            }

        Here:

        1. Same hyperparameters as in
           :class:`~texar.torch.modules.XLNetEncoder`.
           See the :meth:`~texar.torch.modules.XLNetEncoder.default_hparams`.
           An instance of XLNetEncoder is created for feature extraction.

        2. Additional hyperparameters:

            `"regr_strategy"`: str
                The regression strategy, one of:

                - **cls_time**: Sequence-level regression based on the
                  output of the first time step (which is the `CLS` token).
                  Each sequence has a prediction.
                - **all_time**: Sequence-level regression based on
                  the output of all time steps. Each sequence has a prediction.
                - **time_wise**: Step-wise regression, i.e., make
                  regression for each time step based on its output.

            `"logit_layer_kwargs"`: dict
                Keyword arguments for the logit :torch_nn:`Linear` layer
                constructor. Ignored if no extra logit layer is appended.

            `"use_projection"`: bool
                If `True`, an additional :torch_nn:`Linear` layer is added after
                the summary step.

            `"name"`: str
                Name of the regressor.
        """

        hparams = XLNetEncoder.default_hparams()
        hparams.update(({
            "regr_strategy": "cls_time",
            "use_projection": True,
            "logit_layer_kwargs": None,
            "name": "xlnet_regressor",
        }))
        return hparams

    def param_groups(self,
                     lr: Optional[float] = None,
                     lr_layer_scale: float = 1.0,
                     decay_base_params: bool = False):
        r"""Create parameter groups for optimizers. When
        :attr:`lr_layer_decay_rate` is not 1.0, parameters from each layer form
        separate groups with different base learning rates.

        The return value of this method can be used in the constructor of
        optimizers, for example:

        .. code-block:: python

            model = XLNetRegressor(...)
            param_groups = model.param_groups(lr=2e-5, lr_layer_scale=0.8)
            optim = torch.optim.Adam(param_groups)

        Args:
            lr (float): The learning rate. Can be omitted if
                :attr:`lr_layer_decay_rate` is 1.0.
            lr_layer_scale (float): Per-layer LR scaling rate. The `i`-th layer
                will be scaled by `lr_layer_scale ^ (num_layers - i - 1)`.
            decay_base_params (bool): If `True`, treat non-layer parameters
                (e.g. embeddings) as if they're in layer 0. If `False`, these
                parameters are not scaled.

        Returns:
            The parameter groups, used as the first argument for optimizers.
        """

        # TODO: Same logic in XLNetClassifier. Reduce code redundancy.

        if lr_layer_scale != 1.0:
            if lr is None:
                raise ValueError(
                    "lr must be specified when lr_layer_decay_rate is not 1.0")

            fine_tune_group = {
                "params": params_except_in(self, ["_encoder"]),
                "lr": lr
            }
            param_groups = [fine_tune_group]
            param_group = self._encoder.param_groups(lr, lr_layer_scale,
                                                     decay_base_params)
            param_groups.extend(param_group)
            return param_groups
        return self.parameters()

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                segment_ids: Optional[torch.LongTensor] = None,
                input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Feeds the inputs through the network and makes regression.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            segment_ids: Shape `[batch_size, max_time]`.
            input_mask: Float tensor of shape `[batch_size, max_time]`. Note
                that positions with value 1 are masked out.

        Returns:
            Regression predictions.

            - If ``regr_strategy`` is ``cls_time`` or ``all_time``, predictions
              have shape `[batch_size]`.

            - If ``clas_strategy`` is ``time_wise``, predictions have shape
              `[batch_size, max_time]`.
        """
        # output: [batch_size, seq_len, hidden_dim]
        output, _ = self._encoder(inputs=inputs,
                                  segment_ids=segment_ids,
                                  input_mask=input_mask)

        strategy = self._hparams.regr_strategy
        if strategy == 'time_wise':
            summary = output
        elif strategy == 'cls_time':
            summary = output[:, -1]
        elif strategy == 'all_time':
            length_diff = self._hparams.max_seq_length - inputs.shape[1]
            summary_input = F.pad(output, [0, 0, 0, length_diff, 0, 0])
            summary_input_dim = (self._encoder.output_size *
                                 self._hparams.max_seq_length)

            summary = summary_input.contiguous().view(-1, summary_input_dim)
        else:
            raise ValueError('Unknown regression strategy: {}'.format(
                strategy))

        if self._hparams.use_projection:
            summary = torch.tanh(self.projection(summary))

        summary = self.dropout(summary)

        preds = self.hidden_to_logits(summary).squeeze(-1)

        return preds

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output. Since output size is
        only determined by input, the feature size is equal to ``-1``.
        """
        return -1
