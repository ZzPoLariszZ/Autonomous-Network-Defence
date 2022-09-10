from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep, DefenceEvade


class CovertAgent(BaseAgent):
    def __init__(self):
        self.action = 0
        self.target_ip_address = None
        self.last_hostname = None
        self.last_subnet = None
        self.last_ip_address = None
        self.last_observation = None
        self.action_history = {}
        self.jumps = [0,1,2,2,2,2,6,6,6,6,6,11,11,11,11,15,15,16]

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        
        session = 0

        while True:
            if observation['success'] == True:
                self.action += 1 if self.action < 17 else 0
            else:
                self.action = self.jumps[self.action]

            if self.action in self.action_history:
                action = self.action_history[self.action]

            # Discover Remote Systems
            elif self.action == 0:
                self.last_subnet = observation['User0']['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(session=session, agent='Red', subnet=self.last_subnet)

            # Discover Network Services new IP address found
            elif self.action == 1:
                self.last_ip_address = [value for key, value in observation.items() if key != 'success'][1]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

            # ------------------------------------------------------------------------------------------------------------------------------------
            # Exploit User1
            elif self.action == 2:
                 action = ExploitRemoteService(session=session, agent='Red', ip_address=self.last_ip_address)

            # Privilege escalation on User1
            elif self.action == 3:
                self.last_hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=self.last_hostname, session=session)

            # Hide User1 (try to hide the track)
            elif self.action == 4:
                self.last_observation = observation
                action = DefenceEvade(session=session, agent='Red', hostname=self.last_hostname)

            # Discover Network Services new IP address found
            elif self.action == 5:
                self.last_ip_address = self.last_observation['Enterprise1']['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

            # ------------------------------------------------------------------------------------------------------------------------------------
            # Exploit Enterprise1
            elif self.action == 6:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)
            
            # Privilege escalation on Enterprise1
            elif self.action == 7:
                self.last_host = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=self.last_host, session=session)

            # Hide Enterprise1 (try to hide the track)
            elif self.action == 8:
                self.last_observation = observation
                action = DefenceEvade(session=session, agent='Red', hostname='Enterprise1')

            # Scanning the new subnet found.
            elif self.action == 9:
                self.last_subnet = self.last_observation['Enterprise1']['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(subnet=self.last_subnet, agent='Red', session=session)

            # Discover Network Services Enterprise2
            elif self.action == 10:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)

            # ------------------------------------------------------------------------------------------------------------------------------------
            # Exploit Enterprise2
            elif self.action == 11:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

            # Privilege escalation on Enterprise2
            elif self.action == 12:
                self.last_host = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=self.last_host, session=session)

            # Hide Enterprise2 (try to hide the track)
            elif self.action == 13:
                self.target_ip_address = observation['Op_Server0']['Interface'][0]['IP Address']
                action = DefenceEvade(session=session, agent='Red', hostname='Enterprise2')

            # Discover Network Services Op_Server0
            elif self.action == 14:
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)

            # ------------------------------------------------------------------------------------------------------------------------------------
            # Exploit Op_Server0
            elif self.action == 15:
                info = [value for key, value in observation.items() if key != 'success']
                if len(info) > 0:
                    action = ExploitRemoteService(agent='Red', session=session, ip_address=info[0]['Interface'][0]['IP Address'])
                else:
                    self.action = 10
                    continue
            
            # Privilege escalation on Op_Server0
            elif self.action == 16:
                action = PrivilegeEscalate(agent='Red', hostname='Op_Server0', session=session)
            
            # ------------------------------------------------------------------------------------------------------------------------------------
            # Impact on Op_Server0
            elif self.action == 17:
                action = Impact(agent='Red', session=session, hostname='Op_Server0')

            if self.action not in self.action_history:
                self.action_history[self.action] = action
            
            return action

    def end_episode(self):
        self.action = 0
        self.target_ip_address = None
        self.last_hostname = None
        self.last_subnet = None
        self.last_ip_address = None
        self.last_observation = None
        self.action_history = {}

    def set_initial_values(self, action_space, observation):
        pass
