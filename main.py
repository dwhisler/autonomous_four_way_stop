from agent import *
from mdp import *
from util import *

import numpy
import os


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

    total_rewards, visualization = simulate(mdp, qRL)
    visualizer(*visualization)



if __name__ == '__main__':
    main()
