from agent import *
from mdp import *
from util import *

import numpy
import os
import matplotlib.pyplot as plt


def create_grid(n):
    grid = np.zeros((n,n))
    grid[n//2, :]   = 1
    grid[n//2-1, :] = 1
    grid[:, n//2]   = 1
    grid[:, n//2-1] = 1
    return grid

def get_stops(n):
    stops = []
    stops.append((n//2-2, n//2-1))
    stops.append((n//2,   n//2-2))
    stops.append((n//2+1, n//2))
    stops.append((n//2-1, n//2+1))
    return stops

def main():
    # gird size
    n = 6
    # create grid + stops
    grid = create_grid(n)
    stops = get_stops(n)

    mdp = FourWayStopMDP(grid, stops)
    qRL = QLearningAgent(mdp.actions, mdp.discount, identityFeatureExtractor_str, )

    total_rewards, crashes, visualization = simulate(mdp, qRL, maxIterations=10, numTrials=100000)

    def moving_average(a, n=100) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # plt.plot(moving_average(total_rewards))
    # plt.plot(total_rewards)
    plt.plot(moving_average(crashes, 500))
    plt.show()
    # print(list(visualization[0]))

    # qRL.explorationProb = 0
    # total_rewards, visualization = simulate(mdp, qRL, maxIterations=10, numTrials=2)
    # visualizer(*visualization)



if __name__ == '__main__':
    main()
