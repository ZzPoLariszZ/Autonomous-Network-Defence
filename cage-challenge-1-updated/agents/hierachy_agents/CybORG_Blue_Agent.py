import gym, inspect, random
from ray.rllib.env.env_context import EnvContext
from pprint import pprint

from CybORG import CybORG
from CybORG.Agents import GreenAgent, BaseAgent, B_lineAgent, RedMeanderAgent
from CybORG.Agents import CovertAgent, RandomAgent
from CybORG.Agents import BlueMonitorAgent, BlueReactRemoveAgent, BlueReactRestoreAgent
from CybORG.Agents.Wrappers import ChallengeWrapper


class CybORG_Blue_Agent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
        'Red': RandomAgent,  # B_lineAgent, CovertAgent, RedMeanderAgent, RandomAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.steps = 0
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=None):
        random.seed(seed)

