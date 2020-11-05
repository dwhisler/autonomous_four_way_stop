from agent import *
from mdp import *
from util import *

import numpy
import os
import matplotlib.pyplot as plt

################################################################################
# Plotting Helpers
################################################################################
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


################################################################################
# MDP Creation Helpers
################################################################################

def create_grid(n):
    grid = np.zeros((n,n))
    grid[n//2, :]   = 1
    grid[n//2-1, :] = 1
    grid[:, n//2]   = 1
    grid[:, n//2-1] = 1

    # Set wrong lane to value 2
    grid[n//2-1, :n//2-1] = 2
    grid[n//2, n//2+1:] = 2
    grid[n//2+1:, n//2-1] = 2
    grid[:n//2-1, n//2-1] = 2
    return grid

def get_stops(n):
    stops = []
    stops.append((n//2-2, n//2-1))
    stops.append((n//2,   n//2-2))
    stops.append((n//2+1, n//2))
    stops.append((n//2-1, n//2+1))
    return stops


################################################################################
# Main
################################################################################

def main():
    # grid size
    n = 12
    # create grid + stops
    grid = create_grid(n)
    stops = get_stops(n)

    mdp = FourWayStopMDP(grid, stops, num_other=2)
    qRL = QLearningAgent(mdp.actions, mdp.discount, identityFeatureExtractor_str, )

    total_rewards, crashes, visualization = simulate(mdp, qRL, maxIterations=10, numTrials=10000)

    plt.plot(moving_average(total_rewards))
    # plt.plot(total_rewards)
    # plt.plot(total_rewards, label='base')
    plt.plot(smooth(total_rewards, .9), label='.9')
    # plt.plot(smooth(total_rewards, .8), label='.8')
    # plt.plot(smooth(total_rewards, .7), label='.7')
    # plt.plot(smooth(total_rewards, .6), label='.6')
    plt.plot(moving_average(total_rewards, n=20), label='mv 20')
    plt.legend()
    # plt.plot(moving_average(crashes, 500))
    plt.show()
    # print(list(visualization[0]))

    # qRL.explorationProb = 0
    # total_rewards, visualization = simulate(mdp, qRL, maxIterations=10, numTrials=2)
    visualizer(*visualization)



if __name__ == '__main__':
    main()
