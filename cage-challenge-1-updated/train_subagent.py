"""Alternative RLLib model based on local training
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import inspect
import sys

# Ray imports
import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.schedulers import ASHAScheduler # https://openreview.net/forum?id=S1Y7OOlRZ algo for early stopping
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import DEFAULT_CONFIG
import ray.rllib.agents.impala as impala
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import time
# CybORG imports
from CybORG import CybORG
from CybORG.Agents import BaseAgent, GreenAgent, B_lineAgent, RedMeanderAgent
from CybORG.Agents import CovertAgent, RandomAgent
from CybORG.Agents import BlueMonitorAgent, BlueReactRemoveAgent, BlueReactRestoreAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper

from ray.rllib.models.torch.misc import SlimFC

from typing import Any
#import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2

from agents.hierachy_agents.CybORG_Blue_Agent import CybORG_Blue_Agent
from agents.hierachy_agents.CybORG_Red_Agent import CybORG_Red_Agent
from agents.hierachy_agents.curiosity import Curiosity

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
#torch.device(str("cuda:0"))

#gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')    
#tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')

class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)

        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

def normc_initializer(std: float = 1.0) -> Any:
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))

    return initializer

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        #print("model_config", model_config, flush=True)

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space, num_outputs, model_config, name)
        #self.model.to('cuda')
        #super().__init__()

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    adversary_name = CybORG_Blue_Agent.agents['Red'].__name__
    # defender_name = CybORG_Red_Agent.agents['Blue'].__name__
    # defender_name = "Hier"

    print('\033[92m' + "/"*50 + '\033[0m', flush=True)
    print('\033[92m' + "Training defender for " + adversary_name + "..." + '\033[0m', flush=True)
    # print('\033[92m' + "Training attacker for " + defender_name + "..." + '\033[0m', flush=True)
    gpus = 0
    port = 6379
    if len(sys.argv) > 1:
        gpus = int(sys.argv[1])
        port = sys.argv[2]
        print('\033[92m' + "Ray config:   GPUs:" + str(gpus) + " |  Port:" + str(port) + '\033[0m', flush=True)
        ray.init(address='localhost:'+str(port), _redis_password='5241590000000000')    
    else:
        print('\033[92m' + "No params provided. Using default Ray config." + '\033[0m', flush=True)
        ray.init()    
    
    
    print("torch.cuda.is_available()", torch.cuda.is_available())
    torch.device(str("cuda:0"))
    print('\033[92m' + "/"*50 + '\033[0m', flush=True)
    ##############################################################


    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)

    config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,{
        "env": CybORG_Blue_Agent,
        # "env": CybORG_Red_Agent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": gpus, #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_PPO_Model",
            "vf_share_layers": True,
        },
        "lr": 0.0001,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 0,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,
        "vf_clip_param": 5.0,
        "exploration_config": {
            "type": Curiosity,  # <- Use the Curiosity module for exploring.
            "framework": "torch",
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 288,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
                'framework': 'torch',
                'device': 'cuda:0'
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }
    })

    stop = {
        "training_iteration": 100000,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,   # Total number of timesteps
        "episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    log_dir = 'temp_log_dir/'
    if len(sys.argv[1:]) != 1:
        print('No log directory specified, defaulting to: {}'.format(log_dir))
    else:
        log_dir = sys.argv[1]
        print('Log directory specified: {}'.format(log_dir))

    algo = ppo.PPOTrainer
    analysis = tune.run(algo, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        name=algo.__name__ + '_' + adversary_name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        # name=algo.__name__ + '_' + defender_name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        local_dir=log_dir,
                        stop=stop,
                        #restore=checkpoint,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=3,
                        checkpoint_score_attr="episode_reward_min")

    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )

    checkpoint_pointer.write(last_checkpoint)
    print("Best model checkpoint written to: {}".format(last_checkpoint))

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training