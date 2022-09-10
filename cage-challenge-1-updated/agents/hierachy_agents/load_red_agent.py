import os, torch
from pprint import pprint
import os.path as path
import numpy as np
import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

import os
from agents.hierachy_agents.agent_collection import agent_collection
from agents.hierachy_agents.CybORG_Red_Agent import CybORG_Red_Agent

from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

class LoadRedAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self, num_gpus, port) -> None:
        
        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
    
        two_up = path.abspath(path.join(__file__, "../../../"))
        # self.Remove_checkpoint_pointer = two_up + agent_collection['Remove_trained']
        self.Restore_checkpoint_pointer = two_up + agent_collection['Restore_trained']

        # print("Using checkpoint file (Remove): {}".format(self.Remove_checkpoint_pointer))
        print("Using checkpoint file (Restore): {}".format(self.Restore_checkpoint_pointer))

        ray.init(address='localhost:'+str(port), _redis_password='5241590000000000')
        print('torch.device(str("cuda:0")', torch.device(str("cuda:0")))
        print("torch.cuda.is_available()", torch.cuda.is_available())

        config_Remove = {
            "env": CybORG_Red_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0, #int(num_gpus), #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.00005,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.05,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.5,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.1,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }
            }
        }

        config_Restore = {
            "env": CybORG_Red_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0, #int(num_gpus), #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.00005,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.05,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
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
        }

        self.observation = np.zeros((1, 40))

        # load agent trained against BlueRemove
        # self.Remove_attack = ppo.PPOTrainer(config=config_Remove, env=CybORG_Red_Agent)
        # self.Remove_attack.restore(self.Remove_checkpoint_pointer)
        # load agent trained against BlueRestore
        self.Restore_attack = ppo.PPOTrainer(config=config_Restore, env=CybORG_Red_Agent)
        self.Restore_attack.restore(self.Restore_checkpoint_pointer)
        
        ray.shutdown()

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        self.observation = np.roll(self.observation, -1, 0)
        self.observation[0] = obs
        # agent_action = self.Remove_attack.compute_single_action(self.observation[-1:])
        agent_action = self.Restore_attack.compute_single_action(self.observation[-1:])
        
        return agent_action
