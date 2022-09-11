# UCL-2021-2022-COMP0064

Here is the code repository for UCL dissertation project [COMP0064: MSc Information Security Dissertation](https://www.ucl.ac.uk/module-catalogue/modules/msc-information-security-dissertaion/COMP0064)

The selected project topic: Autonomous Network Defence using Reinforcement Learning

Thanks for the help from project supervisors: Dr. Vasilios Mavroudis, Dr. Chris Hicks, and Prof. Steven Murdoch

---
## Setup and Installation

1. Clone the repo using git.

    ``` 
    git clone https://github.com/ZzPoLariszZ/Autonomous-Network-Defence.git
    ```

2. Install the CybORG environment.

    ```
    pip install -e cage-challenge-1/CybORG
    ```

3. Test the environment is installed correctly.

    ```
    pytest cage-challenge-1/CybORG/CybORG/Tests/test_sim
    ```

4. Install the requirements for the dissertation.

    ```
    pip install -r cage-challenge-1-updated/requirements.txt
    ```

5. Before running, make sure all dependencies are correct, and then start ray.

    ```
    ray start --head --address localhost:6379 --redis-password "5241590000000000"
    ```

---
## Repo Structure

### Inside `cage-challenge-1`

- `cage-challenge-1/CybORG/CybORG/Agents/SimpleAgents/`: Directory for red and blue agents

    - `cage-challenge-1/CybORG/CybORG/Agents/SimpleAgents/B_line.py`: B_lineAgent
    
    - `cage-challenge-1/CybORG/CybORG/Agents/SimpleAgents/Covert.py`: CovertAgent

    - `cage-challenge-1/CybORG/CybORG/Agents/SimpleAgents/Meander.py`: MeanderAgent

    - `cage-challenge-1/CybORG/CybORG/Agents/SimpleAgents/Random.py`: RandomAgent

- `cage-challenge-1/CybORG/CybORG/Evaluation/`: Directory for generated evaluation results

- `cage-challenge-1/CybORG/CybORG/Shared/`: Directory for Actions, Scenarios, and Reward Calculator

- `cage-challenge-1/CybORG/CybORG/Tutorial/`: Directory for tutorials of the CAGE challenge

### Inside `cage-challenge-1-unmodified`

- `cage-challenge-1-unmodified/agents/`: Directory for the hierarchical RL method using PPO with ICM

- `cage-challenge-1-unmodified/log_dir/`: Directory for the trained blue agent (controller and subagents)

### Inside `cage-challenge-1-updated`

- `cage-challenge-1-updated/agents/`: Directory for the hierarchical RL method using PPO with ICM

    - `cage-challenge-1-updated/agents/hierachy_agents/agent_collection.py`:

        Path collection for the trained red and blue agents

    - `cage-challenge-1-updated/agents/hierachy_agents/curiosity.py`:

        Code for Intrinsic Curiosity Module

    - `cage-challenge-1-updated/agents/hierachy_agents/CybORG_Red_Agent.py` and

        `cage-challenge-1-updated/agents/hierachy_agents/CybORG_Blue_Agent.py`:

        Environments for training red and blue agents

    - `cage-challenge-1-updated/agents/hierachy_agents/hier_env.py`

        Environment for hierarchical architecture 

    - `cage-challenge-1-updated/agents/hierachy_agents/load_red_agent.py` and
        
        `cage-challenge-1-updated/agents/hierachy_agents/load_blue_agent.py`:

        Code for loading the trained red and blue agents

- `cage-challenge-1-updated/evaluation_results/`: Directory for results, action history, and virtualisation

- `cage-challenge-1-updated/log_dir/`: Directory for the trained red and blue agents

- `cage-challenge-1-updated/temp_log_dir/`: Temporary Directory for the trained red and blue agents (Empty)

---
## Evaluation

### For the trained blue agents (Vanilla PPO)

Run command

```
python3 cage-challenge-1/CybORG/CybORG/Evaluation/evaluation.py
```

### For the trained blue agents (Unmodified)

Optional: select the probability of each red agents in `cage-challenge-1-unmodified/evaluation_hierarchy.py` Line 73

Run command

```
python3 cage-challenge-1-unmodified/evaluation_hierarchy.py
```

### For the trained blue agents (Updated)

Optional: select the probability of each red agents in `cage-challenge-1-updated/evaluation_blue.py` Line 86

Run command

```
python3 cage-challenge-1-updated/evaluation_blue.py
```

### For the trained red agents

Because of compatibility issues, 

please comment all lines associated with the `DefenceEvade` action in the following files:

- `cage-challenge-1/CybORG/CybORG/Agents/SimpleAgents/Covert.py`: Line 4

- `cage-challenge-1/CybORG/CybORG/Shared/RedRewardCalculator.py`: Line 3, 59-65, 68

- `cage-challenge-1/CybORG/CybORG/Shared/Actions/__init__.py`: Line 24

- `cage-challenge-1/CybORG/CybORG/Shared/Actions/AbstractActions/__init__.py`: Line 11

- `cage-challenge-1/CybORG/CybORG/Shared/Scenarios/Scenario1b.yaml`: Line 262

Optional: select the trained red agent in `cage-challenge-1-updated/agents/hierachy_agents/load_red_agent.py`

Optional: select the attacked blue agent in `cage-challenge-1-updated/evaluation_red.py`

Run command

```
python3 cage-challenge-1-updated/evaluation_red.py
```

### Results

All evaluation results will be stored in `cage-challenge-1/CybORG/CybORG/Evaluation/`

---
## Training

### To train blue subagents

Make sure that `adversary_name` and environment `CybORG_Blue_Agent` 

in  `cage-challenge-1-updated/train_subagent.py` is correct

Optional: change the against red agent in `cage-challenge-1-updated/agents/hierachy_agents/CybORG_Blue_Agent.py`

Optional: change hyperparameters in `cage-challenge-1-updated/train_subagent.py`

Run command

```
python3 cage-challenge-1-updated/train_subagent.py
```

### To train blue controller

Optional: change hyperparameters in `cage-challenge-1-updated/train_controller.py`

Run command

```
python3 cage-challenge-1-updated/train_controller.py
```

### To train red agents

Similarly, solve the compatibility issues as before

Make sure that `defender_name` and environment `CybORG_Red_Agent` 

in  `cage-challenge-1-updated/train_subagent.py` is correct

Remove the stop condition in `cage-challenge-1-updated/train_subagent.py` Line 188

Optional: change the against blue agent in `cage-challenge-1-updated/agents/hierachy_agents/CybORG_Red_Agent.py`

Optional: change hyperparameters in `cage-challenge-1-updated/train_subagent.py`

Run command

```
python3 cage-challenge-1-updated/train_subagent.py
```

### Results

All training results will be stored in `cage-challenge-1-updated/temp_log_dir/`

### How to load trained agents

Change paths in the following files to locate the trained agents

- `cage-challenge-1-updated/agents/hierachy_agents/agent_collection.py`

- `cage-challenge-1-updated/agents/hierachy_agents/hier_env.py`

- `cage-challenge-1-updated/agents/hierachy_agents/load_red_agent.py`
        
- `cage-challenge-1-updated/agents/hierachy_agents/load_blue_agent.py`:
