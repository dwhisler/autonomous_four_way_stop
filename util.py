# imports
import numpy as np
import random
import os
from time import sleep
from typing import List, Callable, Tuple, Any

# borrowed from CS221 Homework


# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self) -> Tuple: raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state: Tuple) -> List[Any]: raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state: Tuple, action: Any) -> List[Tuple]: raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print ("%d states" % len(self.states))
        # print (self.states)

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state: Tuple) -> Any: raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple): raise NotImplementedError("Override me")


############################################################


# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp: MDP, rl: RLAlgorithm, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    # visualization
    states = []
    rewards = []
    actions = []
    gridInfo = (mdp.grid, mdp.stops)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            action = rl.getAction(state)

            # visualization
            states.append(state)
            rewards.append(totalReward)
            actions.append(action)

            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print(("Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)))
        totalRewards.append(totalReward)

    visualization = (zip(states, rewards, actions), gridInfo)
    return totalRewards, visualization


def visualizer(results, gridInfo):
    def makeGrid(grid, stops, locations):
        w = len(grid)
        h = len(grid[0])
        ver = [["|   "] * w + ['|'] for _ in range(h)] + [[]]
        hor = [["+———"] * w + ['+'] for _ in range(h + 1)]
     
        def updateHorizontal(x,y):
            if grid[x][y-1] == 0 and grid[x][y] == 0:
                hor[y][x] = f'+ ~ '
            elif grid[x][y] == grid[x][y-1]:
                hor[y][x] = f'+   '   

        def updateVertical(x,y):
            if grid[x-1][y] == grid[x][y]:
                ver[y][x] = f'    '
                return

        def updateStops(stops):
            for x,y in stops:
                if x < w//2 and y < h//2:
                    hor[y+1][x]=u'+\u2013\u2013\u2013'
                elif x < w//2 and y >= h//2:
                    ver[y][x+1]=u'\u2506   '
                elif x >= w//2 and y < h//2:
                    ver[y][x]=u'\u2506   '
                if x >= w//2 and y >= h//2:
                    hor[y][x]=u'+\u2013\u2013\u2013'

        def updateLocations(l, agent=0):
            x,y=l
            if not agent: # our agent
                ver[y][x] = ver[y][x][0]+' Us'
            else: # other agent
                if ver[y][x][1:] == ' Us':
                    ver[y][x] = ver[y][x][0]+u' \u2573 '
                else:
                    ver[y][x] = ver[y][x][0]+'Oth'


        for x in range(w):
            for y in range(h):
                if y > 0:
                    updateHorizontal(x,y)
                if x > 0:
                    updateVertical(x,y)
        updateStops(stops)
        for i, l in enumerate(locations):
            updateLocations(l, i)

     
        s = ""
        for (a, b) in zip(hor, ver):
            s += ''.join(a + ['\n'] + b + ['\n'])
        return s

    def displayStatus(score, action, stand):
        s = ''
        s += u'\u250C\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2510\n'
        s += u'\u2502Score\u2502Action\u2502Stand\u2502\n'
        s += u'\u2502{0:^5}\u2502{1:^6}\u2502{2:^5}\u2502\n'.format(score, actionDict[action], stand)
        s += u'\u2514\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2518\n'
        return s

    clear = lambda: os.system('cls')
    actionDict = {  'north': u'\u2191',
                    'south': u'\u2193',
                    'east': u'\u2192',
                    'west': u'\u2190',
                    'stay': u'\u21BA',}

    grid, stops = gridInfo
    for s, r, a in results:
        clear()
        print(makeGrid(grid=grid, stops=stops, locations=list(s[:-1])),end='')
        print(displayStatus(r,a, s[-1]))
        sleep(.5)

 
if __name__ == '__main__':
    grid = [[0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0]]
    results = [ (((1,1),(2,2),0), 1, 'south'),
                (((1,2),(2,3),0), 2,  'west'),
                (((1,3),(2,4),1), 3,  'north'),
                (((1,4),(1,4),0), -5,  'stay')
            ]
    visualizer(results, (grid, [(2,1), (1,3), (3,4), (4,2)]))
