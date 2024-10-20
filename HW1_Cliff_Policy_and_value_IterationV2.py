import gym
from gym import Env
from gym.spaces import Discrete, Tuple
import random
import numpy as np
import pprint

# import sys
# old_stdout = sys.stdout

# log_file = open("message.log","w",encoding='utf-8')

# sys.stdout = log_file

class CustomEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Tuple((Discrete(4), Discrete(12)))
        # Initialize the grid
        self.grid = np.zeros([4,12])
        self.grid[3][0] = 1
        self.grid[3][11] = 3
        self.max_steps = 15000
        self.step_count = 0
        
        # this could be found from env.unwrapped._cliffs
        self.cliffs = [(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10)]
        
    def step(self, action):
        assert self.action_space.contains(action)

        new_pos = None
        if action == 0: # up
            new_pos = (self.pos[0] - 1, self.pos[1])
        elif action == 1: # right
            new_pos = (self.pos[0], self.pos[1] + 1)
        elif action == 2: # down
            new_pos = (self.pos[0] + 1, self.pos[1])
        elif action == 3: # left
            new_pos = (self.pos[0], self.pos[1] - 1)

        # Make sure the agent doesn't fall off of the grrid
        if self._onGrid(new_pos):
            self.pos = new_pos

        # Check if the agent is in a terminal position
        # Negative reward for every timestep except for terminal step
        done = False
        reward = -1
        if self._inTerminalPos():
            done = True
            reward = 1000

        if self.step_count == self.max_steps:
            done = True

        self.step_count += 1

        return self.pos, reward, done, {}        
    
    def _onGrid(self, pos):
        return pos[0] in range(0,4) and pos[1] in range(0,12)
    
    def _inTerminalPos(self):
        return self.grid[self.pos[0]][self.pos[1]] == 3    
    
    def reset(self):
    	# Reset step count
        self.step_count = 0
        # Start the agent in the top right corner
        self.pos = (3,0)
        return self.pos    
    
    def render(self):
        # Convert every element in the grid from a number to a string
        new_grid = list(
            map(lambda r:
                list(map(lambda c: str(c), r)),
            self.grid)
        )

        # Put an X where the agent currrently is
        new_grid[self.pos[0]][self.pos[1]] = 'X'

        pprint.pprint(new_grid)


class State(object):
    """
    Represents a state or a point in the grid.

    coord: coordinate in grid world
    """
    def __init__(self, coord, is_terminal):
        self.coord = coord
        self.cliffs = [(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10)]
        self.action_state_transitions = self._getActionStateTranstions()
        self.is_terminal = is_terminal
        
        
        self.is_clifs = coord in self.cliffs
        self.reward = -1
        if self.is_clifs:
            self.reward = -100
        if is_terminal:
            self.reward = 1000

        
        
    # Returns a dictionary mapping each action to the following state
    # it would put the agent in from the currrent state
    def _getActionStateTranstions(self):
        action_state_transitions = {}
            
        # Action 0 - up
        if self._isFirstRowState():
            action_state_transitions[0] = self.coord
        else:
            # prev row, same col
            action_state_transitions[0] = (self.coord[0]-1, self.coord[1])

        # Action 1 - right
        if self._isLastColState():
            action_state_transitions[1] = self.coord
        else:
            # same row, next col
            action_state_transitions[1] = (self.coord[0], self.coord[1]+1)

        # Action 2 - down
        if self._isLastRowState():
            action_state_transitions[2] = self.coord
        else:
            # next row, same col
            action_state_transitions[2] = (self.coord[0]+1, self.coord[1])

        # Action 3 - left
        if self._isFirstRowState():
            action_state_transitions[3] = self.coord
        else:
            # same row, prev col
            action_state_transitions[3] = (self.coord[0], self.coord[1]-1)
            
                
        return action_state_transitions

    def _isFirstRowState(self):
        return self.coord[0] == 0

    def _isLastRowState(self):
        return self.coord[0] == 3

    def _isFirstColState(self):
        return self.coord[1] == 0

    def _isLastColState(self):
        return self.coord[1] == 11

    # Returns if the current state is a terminal state
    def isTerminal(self):
        return self.is_terminal

    def is_cliff(self):
        if self.coord in self.cliffs:
            return True
        return False
    
    # Gets the action required to move the agent from the current state
    # to some state s2. If the agent cannot move to s2 it returns None
    def getActionTransiton(self, s2):
        for action, next_state in self.action_state_transitions.items():
            if next_state == s2.coord:
                return action
            if next_state in self.cliffs:
                return None
        return None

    # Returns the likelihood of ending up in state s_prime after taking
    # action a from the current state
    def getNextStateLikelihood(self, a, s_prime):
        if self.action_state_transitions[a] == s_prime.coord:
            return 1
        else:
            return 0

    # Returrn the reward for stepping into this state
    def getReward(self):
        return self.reward
    

class Agent():
    def __init__(self, gamma):
        self.gamma = gamma

        # of states and actions for the grid world problem
        self.num_states = 48
        self.num_actions = 4
        self.cliffs = [(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10)]
        self.pi = None
    # Prints the values of each state on the grid
    def _printStateValues(self, V):
        grid = np.zeros([4,12])

        for state, value in V.items():
            x = state.coord[0]
            y = state.coord[1]
            grid[x,y] = value

        print("Value Function--------------------------")
        pprint.pprint(grid)
        print('\n')
        
    # Prints the policy as a grid of arrows
    def _printPolicy(self, pi):
        grid = np.zeros([4,12])

        for state, actions in pi.items():
            x = state.coord[0]
            y = state.coord[1]
            action = np.argmax(actions)
            grid[x,y] = action

        # Convert actions to arrows
        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, action in enumerate(row):
                arrow_char = 'T'
                if (row_index == 3 and col_index == 11):
                    arrow_grid_row.append(arrow_char)
                elif((row_index,col_index) in self.cliffs):
                    arrow_grid_row.append('C')
                else:
                    if action == 0:
                        arrow_char = '↑'
                    elif action == 1:
                        arrow_char = '→'
                    elif action == 2:
                        arrow_char = '↓'
                    elif action == 3:
                        arrow_char = '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid)
        print('\n')    

    def init_states_valefuntion_policy(self):
        self.S = []
        V = {}
        pi = {}
        for r in range(4):
            for c in range(12):
                # Create the state
                is_terminal = False
                if (r == 3 and c == 11):
                    is_terminal = True
                s = State((r,c,), is_terminal)
                self.S.append(s)
                # Initialize the value of every state to 0
                V[s] = 0
                # Begin with a policy that selects every  action with equal probability
                pi[s] = self.num_actions * [0.25]
        return V, pi

    def getActionValuesForState(self, s, V):
        action_values = []
        for action in range(self.num_actions):
            action_value = 0
            for s_prime in self.S:
                p = s.getNextStateLikelihood(action, s_prime)
                if(p == 1):
                    print("S pos:",s.coord, " s_ pos: ",s_prime.coord, " act: ",action, "rew: ",s_prime.getReward())
                action_value += p * (s_prime.getReward() + self.gamma * V[s_prime])
            action_values.append(action_value)
        return action_values
    
    def play_with_policy(self,state):
        state = self.S[state]    
        action = np.argmax(self.pi[state])
        return action
    
class PolicyIteration(Agent):
    def __init__(self, gamma):
        # Call base class constructor
        super().__init__(gamma)
        self.pi = None
    def policyIterate(self):
        V, pi = self.init_states_valefuntion_policy()

        policy_stable = False
        i = 1
        while not policy_stable:
            print("Policy Iteration", i)
            V = self._iterPolicyEval(pi, V)
            pi, V, policy_stable = self.policyImprove(pi, V)
            self._printPolicy(pi)
            i += 1
        self.pi = pi
    def _iterPolicyEval(self, pi, V):
        # threshold for determing the end of the policy eval loop
        theta = 0.000001

        while True:
            # change in state values
            delta = 0
            total_reward_for_episode = 0
            for s in self.S:
                # current state value
                v = V[s]

                # set the new state value
                V[s] = 0

                if s.isTerminal():
                    continue

                for s_prime in self.S:
                    p = self._p(s, s_prime, pi)
                    V[s] += p * (s_prime.getReward() + self.gamma * V[s_prime])
                    total_reward_for_episode += s_prime.getReward()
                delta = max(delta, abs(v - V[s]))

            self._printStateValues(V)
            print("Total Reward for episode: ",total_reward_for_episode)
            if delta < theta:
                break
        
        return V

    def _p(self, s, s_prime, pi):
        # Get the action that would take the agent from s to s_prime
        transition_action = s.getActionTransiton(s_prime)

        # if the agent cannot move from s to s_prime it would never be selected
        if transition_action == None:
            return 0

        # return probability of selecting this action under the policy
        return pi[s][transition_action]

    def policyImprove(self, pi, V):
        policy_stable = True

        for s in self.S:
            old_best_action = np.argmax(pi[s])
            # print(old_action)

            if s.isTerminal():
                continue

            action_values = self.getActionValuesForState(s, V)

            new_best_action = np.argmax(action_values)

            # Set the likelihood of selecting the new best action in the policy to 1
            # for all other actions make it 0
            for action in range(self.num_actions):
                if action != new_best_action:
                    pi[s][action] = 0
                else:
                    pi[s][action] = 1

            if old_best_action != new_best_action:
                policy_stable = False

        return pi, V, policy_stable
    


class ValueIteration(Agent):
    def __init__(self, gamma):
        # Call base class constructor
        super().__init__(gamma)
        self.pi = None    
    def valueIterate(self):
        V, pi = self.init_states_valefuntion_policy()

        # threshold for determing the end of the policy eval loop
        theta = 0.01

        i = 1
        while True:
            print("Value Iteration", i)
            # change in state values
            delta = 0

            for s in self.S:
                # current state value
                v = V[s]

                # set the new state value
                V[s] = 0

                if s.isTerminal():
                    continue

                action_values = self.getActionValuesForState(s, V)

                V[s] = max(action_values)
                new_best_action = np.argmax(action_values)

                # Set the likelihood of selecting the new best action in the policy to 1
                # for all other actions make it 0
                for action in range(self.num_actions):
                    if action != new_best_action:
                        pi[s][action] = 0
                    else:
                        pi[s][action] = 1

                delta = max(delta, abs(v - V[s]))

            self._printStateValues(V)

            if delta < theta:
                break

            i += 1

        self._printPolicy(pi)    
        self.pi = pi
    

env = CustomEnv()
env.reset()

# pi_agent = PolicyIteration(0.9)
# pi_agent.policyIterate()


vi_agent = ValueIteration(0.9)
vi_agent.valueIterate()




env = gym.make('CliffWalking-v0',render_mode = "human")
state,_ = env.reset()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
terminated = 0
truncated = 0
while not (terminated or truncated):
    env.render()
    action = vi_agent.play_with_policy(state)
    state,raward,terminated,truncated,info =  env.step(action)
    print(state, "---->", action)
    
    # print(env.render())
    



    
# sys.stdout = old_stdout

# log_file.close()