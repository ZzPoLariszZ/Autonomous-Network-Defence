import os.path as path
import gym, inspect, os, random
from CybORG import CybORG
from gym import spaces
import numpy as np
import torch
import ray.rllib.agents.ppo as ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from CybORG.Agents import BaseAgent, GreenAgent, B_lineAgent, CovertAgent, RedMeanderAgent, RandomAgent
from CybORG.Agents.Wrappers import ChallengeWrapper


#from agents.hierachy_agents.scaffold_env import *
from agents.hierachy_agents.agent_collection import agent_collection
from agents.hierachy_agents.CybORG_Blue_Agent import CybORG_Blue_Agent

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


class HierEnv(gym.Env):
    # Env parameters
    max_steps = 100 # Careful! There are two other envs!
    mem_len = 1

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):

        self.cyborg = CybORG(self.path, 'sim', agents={'Red':B_lineAgent})
        self.BAenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')

        self.cyborg = CybORG(self.path, 'sim', agents={'Red':CovertAgent})
        self.CAenv = ChallengeWrapper(env=self.cyborg, agent_name='Blue')

        self.cyborg = CybORG(self.path, 'sim', agents={'Red':RedMeanderAgent})
        self.MAenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        
        self.cyborg = CybORG(self.path, 'sim', agents={'Red':RandomAgent})
        self.RAenv = ChallengeWrapper(env=self.cyborg, agent_name='Blue')


        two_up = path.abspath(path.join(__file__, "../../.."))
        self.BA_checkpoint_pointer = two_up + agent_collection['B_line_trained']
        self.CA_checkpoint_pointer = two_up + agent_collection['Covert_trained']
        self.MA_checkpoint_pointer = two_up + agent_collection['Meander_trained']
        self.RA_checkpoint_pointer = two_up + agent_collection['Random_trained']
    
        
        sub_config_BA = {
            "env": CybORG_Blue_Agent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
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
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
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
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
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
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": False,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
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

        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)

        # Restore the checkpointed model
        self.BA_def = ppo.PPOTrainer(config=sub_config_BA, env=CybORG_Blue_Agent)
        self.CA_def = ppo.PPOTrainer(config=sub_config_CA, env=CybORG_Blue_Agent)
        self.MA_def = ppo.PPOTrainer(config=sub_config_MA, env=CybORG_Blue_Agent)
        self.RA_def = ppo.PPOTrainer(config=sub_config_RA, env=CybORG_Blue_Agent)
        
        self.BA_def.restore(self.BA_checkpoint_pointer)
        self.CA_def.restore(self.CA_checkpoint_pointer)
        self.MA_def.restore(self.MA_checkpoint_pointer)
        self.RA_def.restore(self.RA_checkpoint_pointer)

        self.steps = 0
        self.agent_name = 'BlueHier'

        # action space is 4 for each trained agent to select from
        self.action_space = spaces.Discrete(4)

        # observations for controller is a sliding window of 4 observations
        self.observation_space = spaces.Box(-1.0,1.0,(self.mem_len,52), dtype=float)

        # default observation is 4 lots of nothing
        self.observation = np.zeros((self.mem_len,52))

        self.action = None
        self.env = self.BAenv

    # reset doesnt reset the sliding window of the agent so it can differentiate between
    # agents across episode boundaries
    def reset(self):
        self.steps = 0
        #rest the environments of each attacker
        self.BAenv.reset()
        self.CAenv.reset()
        self.MAenv.reset()
        self.RAenv.reset()
        r = random.choice([0,1,2,3])
        if r == 0:
            self.env = self.BAenv
        elif r == 1:
            self.env = self.CAenv
        elif r == 2:
            self.env = self.MAenv
        else:
            self.env = self.RAenv
        return np.zeros((self.mem_len,52))

    def step(self, action=None):
        # select agent
        if action == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BA_def.compute_single_action(self.observation[-1:])
        elif action == 1:
            # get action from agent trained against the CovertAgent
            agent_action = self.CA_def.compute_single_action(self.observation[-1:])
        elif action == 2:
            # get action from agent trained against the MeanderAgent
            agent_action = self.MA_def.compute_single_action(self.observation[-1:])
        elif action == 3:
            # get action from agent trained against the RandomAgent
            agent_action = self.RA_def.compute_single_action(self.observation[-1:])
        else:
            print('something went terribly wrong, old sport')
        observation, reward, done, info = self.env.step(agent_action)

        # update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[self.mem_len-1] = observation      # Replace what's on the rightmost position

        self.steps += 1
        if self.steps == self.max_steps:
            return self.observation, reward, True, info
        assert(self.steps <= self.max_steps)
        result = self.observation, reward, done, info
        return result

    def seed(self, seed=None):
        random.seed(seed)
