# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import distributions as pyd


class FixedCategorical(pyd.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_prob(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(pyd.Normal):
    def __init__(self, loc, scale) -> None:
        super().__init__(loc, scale)

    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    @property
    def mode(self):
        return self.loc
