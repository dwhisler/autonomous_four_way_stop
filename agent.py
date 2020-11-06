import util
from typing import List, Callable, Tuple, Any
from collections import defaultdict
import random
import math
import numpy as np

class QLearningAgent(util.RLAlgorithm):
    def __init__(self, actions: Callable, discount: float, featureExtractor: Callable, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state: Tuple, action: Any) -> float:
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: Tuple) -> Any:
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            m = max((self.getQ(state, action), action) for action in self.actions(state))[0]
            possible = [a for a in self.actions(state) if self.getQ(state, a) == m]
            sorted(possible)
            return possible[0]
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        # return .1
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple) -> None:
        # helpful numbers
        eta = self.getStepSize()
        gamma = self.discount()
        # get Q(s,a) 
        Q = self.getQ(state, action)
        # initialize
        V_opt = 0
        # if not in an endstate
        if newState:
            # V_opt = self.getQ(newState, self.getAction(newState))
            V_opt = max([self.getQ(newState,newAction) for newAction in self.actions(newState)])

        # print(1, state, action, reward, newState)
        # print(2, eta, gamma, V_opt, Q)
        for f, v in self.featureExtractor(state,action):
            self.weights[f] -= eta*(Q - (reward + gamma*V_opt))*v


# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state: Tuple, action: Any) -> List[Tuple[Tuple, int]]:
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization. 
# Casts feature key to a string
def identityFeatureExtractor_str(state: Tuple, action: Any) -> List[Tuple[Tuple, int]]:
    featureKey = str((state, action))
    featureValue = 1
    return [(featureKey, featureValue)]


# Custom feature extractor. Returns list of (feature, value) pairs
def customFeatureExtractor(state: Tuple, action: Any, n: int) -> List[Tuple[Tuple, int]]:
    # distance helper
    def dist(p1, p2, norm=1):
        if norm == 1:
            return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    # direction finder
    def dir(p1, p2):
        yDif = p1[0] - p2[0]
        xDif = p1[0] - p2[0]
        if abs(xDif) > abs(yDif): # care about y dif then
            if yDif > 0:
                return 'north'
            else:
                return 'south'
        else:
            if xDif > 0:
                return 'west'
            else:
                return 'east'

    # initialize features
    features = []

    # extract state components
    agent, otherAgents, counter = state
    agent = tuple(agent)

    # add current location and action
    features.append((('agent_loc', agent, action),1))

    # add new dist to goal
    actionDict = {'north':np.array([-1,0]),
                    'south':np.array([1,0]),
                    'west':np.array([0,-1]),
                    'east':np.array([0,1]),
                    'stay':np.array([0,0])}
    newAgent = np.array(agent) + actionDict[action]
    features.append((('newDist2Goal', dist(newAgent, (0, int(n/2)))),1))

    # distance to closest agent
    minDist = float('inf')
    minDir = ''
    for otherAgent in otherAgents:
        tempDist = dist(newAgent, otherAgent)
        if tempDist < minDist:
            minDist = tempDist
            minDir = dir(agent, otherAgent)
    features.append((('closestOther', minDist), 1))
    features.append((('dirClosestOther', minDir, action), 1))

    # CLOSE
    features.append((('otherClose', minDist < 3),1))


    # add current counter and location
    features.append((('counter', agent, counter, action), 1))


    # add other locations with distance
    for otherAgent in otherAgents:
        # features.append((('otherAgent', dist(agent, otherAgent), action),1))
        # crash
        if np.array_equal(otherAgent, newAgent):
            features.append(('crash', -1)) # increasing this number decreases crash percentage but also greatly hurts score

    return features