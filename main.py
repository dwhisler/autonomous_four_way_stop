from agent import *
from mdp import *
from util import *

import numpy
import os
import matplotlib.pyplot as plt
import argparse

################################################################################
# Arguments
################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--train' , '-tr', action='store', type=int, 
                            default=1000, help='Number of training trials')
parser.add_argument('--test'  , '-ts', action='store', type=int, 
                            default=100, help='Number of test trials')
parser.add_argument('--iters' , '-i' , action='store', type=int, 
                            default=10, help='Max iterations per trial')
parser.add_argument('--size'  , '-n' , action='store', type=int, 
                            default=6, help='Grid size: nxn')
parser.add_argument('--others', '-o' , action='store', type=int, 
                            default=2, help='Number of other agents')
parser.add_argument('--plot'  , '-p' , action='store_true', 
                            default=False, help='Plotting Flag')
parser.add_argument('--viz'   , '-v' , action='store_true', 
                            default=False, help='Visualization Flag')
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

def main(args):
    ##### Initialization #####
    # grid size
    n = args.size
    # create grid + stops
    grid = create_grid(n)
    stops = get_stops(n)

    # create custom feature extractor
    nFeatureExtractor = lambda s, a: customFeatureExtractor(s, a, n=n)

    # create MDP
    mdp = FourWayStopMDP(grid, stops, num_other=args.others)
    # create Q Learning
    qRL = QLearningAgent(mdp.actions, mdp.discount, nFeatureExtractor)  # identityFeatureExtractor_str

    ##### Training #####
    total_rewards, crashes, visualization = simulate(mdp, qRL, 
                                                        maxIterations=args.iters, 
                                                        numTrials=args.train)


    ##### Plotting #####
    if args.plot:
        plt.plot(moving_average(total_rewards, 50), label='avg 50')
        plt.plot(total_rewards, linewidth=.5, label='raw')
        plt.legend()
        plt.show()
        
        plt.plot(moving_average(crashes, 50), label='avg 50')
        plt.plot(crashes, linewidth=.5, label='raw')
        plt.legend()
        plt.show()

    ##### Testing #####
    qRL.explorationProb = 0
    total_rewards, crashes, visualization = simulate(mdp, qRL, 
                                                        maxIterations=args.iters, 
                                                        numTrials=args.test)
    if args.viz:
        visualizer(visualization, (grid, stops))
    print(f'Average Test Reward: {np.mean(total_rewards)}')
    print(f'Num Test Crashes: {sum(crashes)} - {np.mean(crashes)*100:.3f}%')



if __name__ == '__main__':
    main(parser.parse_args())
