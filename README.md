# BC-IRL: Learning Generalizable Reward Functions from Demonstrations

This is the official PyTorch code implementation for ["BC-IRL: Learning Generalizable Reward Functions from Demonstrations"](https://arxiv.org/abs/2303.16194).


<p align="center">
    <img width="85%" src="https://github.com/ASzot/bcirl/raw/main/title.png">
    <br />
    <a href="https://arxiv.org/abs/2303.16194">[Paper]</a>
</p>

## Installation
* Requires Python >= 3.7: `conda create -y -n bcirl python=3.7`
* `pip install -e .`

## Commands
To run the obstacle version, substitute `pointmass` with `pointmass_obstacle`.

### Train Rewards
* BC-IRL-PPO `python imitation_learning/run.py +bc_irl=pointmass`
* GCL `python imitation_learning/run.py +gcl=pointmass`
* AIRL `python imitation_learning/run.py +airl=pointmass`
* MaxEnt `python imitation_learning/run.py +maxent=pointmass`

### Evaluate Rewards
To evaluate on the `eval` distribution add: `env=pointmass_eval`. Specify the path of the saved reward with `load_checkpoint=`.
* BC-IRL-PPO `python imitation_learning/eval.py +meta_irl=pointmass load_checkpoint=saved_reward.pth`
* GCL `python imitation_learning/eval.py +gcl=pointmass load_checkpoint=saved_reward.pth`
* AIRL `python imitation_learning/eval.py +airl=pointmass load_checkpoint=saved_reward.pth`
* MaxEnt `python imitation_learning/eval.py +maxent=pointmass load_checkpoint=saved_reward.pth`

## Code Structure
Structure of the code under `imitation_learning`:
* `run.py`: Code for the main training loop. 
* `config`: The `yaml` config files for Hydra split by each method. Under `config/env` are the configs for the different environment settings (such as the generalization setting). `config/logger` contains the configs for the [WandB](https://wandb.ai) and CLI logger. `config/default.yaml` contains the default settings shared across all methods. 
* `policy_opt`: Code for the policy and PPO updater. The PPO updater is designed to be differentiable with respect to the rewards for use in BC-IRL.
* `bcirl`: The BC-IRL method
* `gail`: The GAIL baseline. 
* `gcl`: The GCL baseline.
* `maxent`: The MaxEntIRL baseline.
* `common`: Utilities for plotting in the point mass navigation task, reward functions, and other helper functions.

# Citation
```
@article{szot2023bc,
  title={BC-IRL: Learning Generalizable Reward Functions from Demonstrations},
  author={Szot, Andrew and Zhang, Amy and Batra, Dhruv and Kira, Zsolt and Meier, Franziska},
  journal={arXiv preprint arXiv:2303.16194},
  year={2023}
}
```

# License
this code is licensed under the CC-BY-NC license, see LICENSE.md for more details
