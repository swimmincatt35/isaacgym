# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
from datetime import datetime

# noinspection PyUnresolvedReferences
import isaacgym

import hydra

import sys

# import error include parent path needed
sys.path.append("../")
from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank

from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """
    
    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled # Default False
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier'] # Default key does not exist
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass
    
    return config_dict

# [CH] Entry point read cfg/config.yaml
@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    if cfg.pbt.enabled:
        print("enabled")
        initial_pbt_check(cfg)
    
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv

    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    # https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/model_builder.py
    # Global variable NETWORK_REGISTRY = {} and MODEL_REGISTRY = {}


    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs


    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):

        # [CH] init task, ie, CartpoleSample
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, # Default 512
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video, # Default False
            cfg.force_render, # Default True
            cfg,
            **kwargs,
        ) 
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]

    # [CH] hasattr(ige_env_cls, 'dict_obs_cls'): False
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False
    
    if dict_cls:
        
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:
        # [CH] num_actors same as num_envs
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train) 
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict) # [CH] only displaying the train configs

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate: # [CH] Default False
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    # https://github.com/Denys88/rl_games/blob/master/rl_games/torch_runner.py
    def build_runner(algo_observer):
        runner = Runner(algo_observer)

        # Override algo_factory + player_factory
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))

        # Global variable: 
        # [CH] register a model that can be initialized by network and **kwargs
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        # [CH] register a network that can be initialized by **kwargs
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers)) # [CH] Meta Observer wrapper, default only got RLGPUAlgoObserver()
    
    # [CH] deepcopy config['params'], init by loading (multi/single)gpu, random seed, algos, etc.
    runner.load(rlg_config_dict) 
    # [CH] literally does nothing
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        # [CH] ./runs/Cartpole_23-15-26-27
        os.makedirs(experiment_dir, exist_ok=True)  
        # [CH] ./runs/Cartpole_23-15-26-27/config.yaml
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))         

    if cfg.test:
        # [CH] Set player configs
        player_config = {}
        player_config["games_num"] = 10
        # player_config["max_steps"] = 600 <- [CH] This was hard coded :( https://github.com/Denys88/rl_games/blob/66ce12f30f2582d43c818356baba1812669841db/rl_games/common/player.py#L76 
        runner.params['config']['player'] = player_config


    # [CH] cfg written in ./runs/Cartpole_23-15-26-27/config.yaml file
    # [CH] ckpt pipeline for both train and play, 
    # [CH] default algo is a2c_continuous: https://github.com/Denys88/rl_games/blob/66ce12f30f2582d43c818356baba1812669841db/rl_games/algos_torch/a2c_continuous.py#L70
    runner.run({
        'train': not cfg.test,          # [CH] agent.train(), https://github.com/Denys88/rl_games/blob/66ce12f30f2582d43c818356baba1812669841db/rl_games/common/a2c_common.py#L1290 
        'play': cfg.test,               # [CH] player.run(), https://github.com/Denys88/rl_games/blob/66ce12f30f2582d43c818356baba1812669841db/rl_games/common/player.py#L273 
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

    


if __name__ == "__main__":
    launch_rlg_hydra()
