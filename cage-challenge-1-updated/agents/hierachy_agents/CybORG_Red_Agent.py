import gym, inspect, random
from ray.rllib.env.env_context import EnvContext
from pprint import pprint

from CybORG import CybORG
from CybORG.Agents import BaseAgent, GreenAgent, B_lineAgent, RedMeanderAgent
from CybORG.Agents import CovertAgent, RandomAgent
from CybORG.Agents import BlueReactRemoveAgent, BlueReactRestoreAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper

class CybORG_Red_Agent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
        'Blue': BlueReactRestoreAgent, # BlueReactRemoveAgent, BlueReactRestoreAgent
    }
    
    # agents = BlueLoadAgent()

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Red')
        self.steps = 0
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        obs, reward, done, info = self.env.step(action=action)
        reward = self.env.get_rewards()["Red"]
        result = obs, reward, done, info
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=None):
        random.seed(seed)

