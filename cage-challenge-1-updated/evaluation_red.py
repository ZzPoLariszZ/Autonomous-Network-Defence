import inspect
import time
import numpy as np
from statistics import mean, stdev
from pprint import pprint

from CybORG import CybORG
from CybORG.Agents import SleepAgent, B_lineAgent, RedMeanderAgent, CovertAgent, RandomAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent, BlueReactRestoreAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from agents.hierachy_agents.load_red_agent import LoadRedAgent

MAX_EPS = 100
agent_name = 'Red'

randomagents = [BlueReactRestoreAgent]

def wrap(env):
    return OpenAIGymWrapper(env=env, agent_name='Blue')


if __name__ == "__main__":
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    # ask for a name
    name = 'Mindrake ' #input('Name: ')
    # ask for a team
    team = 'Mindrake' #input("Team: ")
    # ask for a name for the agent
    name_of_agent = 'RLLIB PPO Hier' #input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    gpus = 0
    port = 6379
    agent = LoadRedAgent(gpus, port)

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    rew_total = 0

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    #for num_steps in [30, 50, 100]:
    for num_steps in [100]:
        
        total_reward = []
        actions = []
        for i in range(MAX_EPS):
            r = []
            a = []

            # Replicate the HierEnv by randomly choosing a red agent
            #for red_agent in :
            blue_agent = np.random.choice(randomagents, p=[1])
            # agent.set_red_agent(red_agent)
            cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
            wrapped_cyborg = ChallengeWrapper(env=cyborg, agent_name='Red') #wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)

            # cyborg.env.env.tracker.render()
            for j in range(num_steps):
                # blue_obs = cyborg.get_observation('Red')
                # pprint(blue_obs)
                # print('-'*76)
                action = agent.get_action(observation, action_space)
                observation, rew, done, info = wrapped_cyborg.step(action)
                # result = cyborg.step(agent_name, action)

                # Print true table on each step
                #true_state = cyborg.get_agent_state('True')
                #true_table = true_obs_to_table(true_state,cyborg)
                #print(true_table)

                r.append(rew)
                # r.append(result.reward)
                #agent_selected = 'BLine' if agent_selected == 0 else 'RedMeander'
                a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
            total_reward.append(sum(r))
            actions.append(a)
            # observation = cyborg.reset().observation
            observation = wrapped_cyborg.reset()
            #agent.reset()
        print(f'Average reward vs random (P=0.5) red agent and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
        with open(file_name, 'a+') as data:
            data.write(f'steps: {num_steps}, adversary: {blue_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
            for act, sum_rew in zip(actions, total_reward):
                data.write(f'actions: {act}, total reward: {sum_rew}\n')
        rew_total += mean(total_reward)
    print('Total final score: {:.2f}'.format(rew_total))