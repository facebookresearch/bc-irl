# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import OmegaConf

from imitation_learning.run import main


@hydra.main(config_path="config", config_name="default")
def run_and_eval(cfg):
    eval_cfg = OmegaConf.merge(cfg, cfg.eval_args)
    eval_cfg.load_policy = False
    assert eval_cfg.load_checkpoint != "" and eval_cfg.load_checkpoint is not None

    print("Evaluating reward function")
    main(eval_cfg)


if __name__ == "__main__":
    run_and_eval()
