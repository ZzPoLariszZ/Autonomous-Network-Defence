import os, torch
from pprint import pprint
import os.path as path
import numpy as np
import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from agents.hierachy_agents.hier_env import HierEnv, TorchModel
import os
from CybORG.Agents import SleepAgent, B_lineAgent, CovertAgent, RedMeanderAgent, RandomAgent
from agents.hierachy_agents.agent_collection import agent_collection
from agents.hierachy_agents.CybORG_Blue_Agent import CybORG_Blue_Agent
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

class LoadBlueAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self, num_gpus, port) -> None:
        ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)

        # Load checkpoint locations of controller and each agent
        two_up = path.abspath(path.join(__file__, "../../../"))
        self.CTRL_checkpoint_pointer = two_up + agent_collection['Controller_trained']
        self.BA_checkpoint_pointer = two_up + agent_collection['B_line_trained']
        self.CA_checkpoint_pointer = two_up + agent_collection['Covert_trained']
        self.MA_checkpoint_pointer = two_up + agent_collection['Meander_trained']
        self.RA_checkpoint_pointer = two_up + agent_collection['Random_trained']

        print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BA_checkpoint_pointer))
        print("Using checkpoint file (Covert): {}".format(self.CA_checkpoint_pointer))
        print("Using checkpoint file (Meander): {}".format(self.MA_checkpoint_pointer))
        print("Using checkpoint file (Random): {}".format(self.RA_checkpoint_pointer))

        ray.init(address='localhost:'+str(port), _redis_password='5241590000000000')
        print('torch.device(str("cuda:0")', torch.device(str("cuda:0")))
        print("torch.cuda.is_available()", torch.cuda.is_available())

        config = Trainer.merge_trainer_configs(
            DEFAULT_CONFIG,
            {
            "env": HierEnv,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(num_gpus),
            "model": {
                "custom_model": "CybORG_hier_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            #"momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
             "in_evaluation": True,
            'explore': False
        })

        # Restore the controller model
        self.controller_agent = ppo.PPOTrainer(config=config, env=HierEnv)
        self.controller_agent.restore(self.CTRL_checkpoint_pointer)
        self.observation = np.zeros((HierEnv.mem_len,52))

        sub_config_BA = {
            "env": CybORG_Blue_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(num_gpus), #int(num_gpus), #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
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

        sub_config_CA = {
            "env": CybORG_Blue_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(num_gpus), #int(num_gpus), #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
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

        sub_config_MA = {
            "env": CybORG_Blue_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(num_gpus), #int(num_gpus), #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": False,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
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

        sub_config_RA = {
            "env": CybORG_Blue_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(num_gpus), #int(num_gpus), #int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": False,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
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

        #load agent trained against B_lineAgent
        self.BA_def = ppo.PPOTrainer(config=sub_config_BA, env=CybORG_Blue_Agent)
        self.BA_def.restore(self.BA_checkpoint_pointer)
        #load agent trained against CovertAgent
        self.CA_def = ppo.PPOTrainer(config=sub_config_CA, env=CybORG_Blue_Agent)
        self.CA_def.restore(self.CA_checkpoint_pointer)
        #load agent trained against MeanderAgent
        self.MA_def = ppo.PPOTrainer(config=sub_config_MA, env=CybORG_Blue_Agent)
        self.MA_def.restore(self.MA_checkpoint_pointer)
        #load agent trained against RandomAgent
        self.RA_def = ppo.PPOTrainer(config=sub_config_RA, env=CybORG_Blue_Agent)
        self.RA_def.restore(self.RA_checkpoint_pointer)
        
        self.red_agent = -1
        
        ray.shutdown()


    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[HierEnv.mem_len-1] = obs           # Replace what's on the rightmost position

        #select agent to compute action
        if self.red_agent == B_lineAgent: # or self.red_agent == SleepAgent:
            agent_to_select = 0
        elif self.red_agent == CovertAgent:
            agent_to_select = 1
        elif self.red_agent == RedMeanderAgent:
            agent_to_select = 2
        else: # RandomAgent
            agent_to_select = 3

        if agent_to_select == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BA_def.compute_single_action(self.observation[-1:])
        elif agent_to_select == 1:
            # get action from agent trained against the CovertAgent
            agent_action = self.CA_def.compute_single_action(self.observation[-1:])
            # agent_action = self.EDRM_def.compute_single_action(self.observation[-1:])
        elif agent_to_select == 2:
            # get action from agent trained against the MeanderAgent
            agent_action = self.MA_def.compute_single_action(self.observation[-1:])
        elif agent_to_select == 3:
            # get action from agent trained against the RandomAgent
            agent_action = self.RA_def.compute_single_action(self.observation[-1:])
        
        return agent_action
