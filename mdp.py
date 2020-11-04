import util
import numpy as np
import random
import itertools
from typing import List, Callable, Tuple, Any, NewType

# taken from CS 221 HW4

# TODO: Randomize start locations, add other agents

class FourWayStopMDP(util.MDP):
    def __init__(self, grid: np.ndarray, stops: List[Tuple], num_other=1):
        """
        grid: 2D numpy array which maps the initial road layout
        stops: List of (x,y) for where stop signs are located
                -> alternatively, this could be incorporated into grid
        """
        self.grid = grid # grid oriented so that (0,0) is top left corner
        self.stops = stops
        self.action_dict = {'north':np.array([-1,0]),
                            'south':np.array([1,0]),
                            'west':np.array([0,-1]),
                            'east':np.array([0,1]),
                            'stay':np.array([0,0])}
        self.dest = None
        self.other = [None for _ in range(num_other)]
        self.stopped = False
        self.n = self.grid.shape[0]

        # self.debug = True
        # self.debug_count = 0

    # Return the start state.
    def startState(self) -> Tuple:
        # State comes in form (agent location (2-ndarray), other location (2-ndarray), stay counter)
        # Assumption: nxn grid where n is even
        n = self.n
        north_start = (0, int(n/2) - 1)
        north_end = (0, int(n/2))
        south_start = (n-1, int(n/2))
        south_end = (n-1, int(n/2) - 1)
        west_start = (int(n/2), 0)
        west_end = (int(n/2) - 1, 0)
        east_start = (int(n/2) - 1, n-1)
        east_end = (int(n/2), n-1)

        agent_loc = random.choice([south_start, west_start, east_start])
        self.dest = np.array((0, int(n/2))) #np.array(random.choice([north_end, west_end, east_end]))
        self.starts = [north_start, south_start, west_start, east_start]
        self.starts.remove(agent_loc)
        ends = [north_end, south_end, west_end, east_end]
        # Chooses random start location and end location, where start != end

        for k in range(len(self.other)):
            if len(self.starts) == 0:
                self.starts = [north_start, south_start, west_start, east_start]
                self.starts.remove(agent_loc)
            other_loc = random.choice(self.starts)
            self.starts.remove(other_loc)
            other_loc = np.array(other_loc)
            other_dest = np.array(random.choice(ends))
            while (np.linalg.norm(other_dest-other_loc)) <= 1: # if choose end on same side as start
                other_dest = np.array(random.choice(ends))

            self.other[k] = OtherActor(start=other_loc, dest=other_dest, probstop=0.8, gridsz=self.n)

        stay_counter = 0

        return (np.array(agent_loc), [other.start for other in self.other], stay_counter)

    # Return set of actions possible from |state|.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state: Tuple) -> List[str]:
        actions = ['stay']
        agent_loc = state[0]
        if agent_loc[0] > 0:
            actions.append('north')
        if agent_loc[0] < self.n-1:
            actions.append('south')
        if agent_loc[1] > 0:
            actions.append('west')
        if agent_loc[1] < self.n-1:
            actions.append('east')
        return actions

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after accident or arriving)
    #     possibly by setting the flag?
    # * If |state| is an end state, you should make our location None (or just return None) to indicate, and have appropriate reward for crash or goal
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.

    def succAndProbReward(self, state: Tuple, action: str) -> List[Tuple]:
        agent_loc = state[0]
        other_locs = state[1]
        stay_counter = state[2]

        if np.array_equal(agent_loc, self.dest): # if at goal
            return []
        for other_loc in other_locs:
            if np.array_equal(agent_loc, other_loc): # if crashed
                return []

        transitions = []
        other_action_probs = []
        for k in range(len(self.other)):
            other_action_probs.append(self.other[k].get_action_probs(other_locs[k], self.grid, self.stops))

        agent_loc_new = agent_loc + self.action_dict[action]

        # lemcardenas: mod each coordinate or else we go off the grid
        # agent_loc_new[0] %= 6
        # agent_loc_new[1] %= 6

        if np.array_equal(agent_loc_new, agent_loc):
            stay_counter += 1
        else:
            stay_counter = 0

        for action_prob_combination in itertools.product(*other_action_probs):
            other_actions = [combo[0] for combo in action_prob_combination]
            other_probs = [combo[1] for combo in action_prob_combination]
            # print(other_actions)
            # print(other_probs)
            other_locs_new = [(other_locs[k] + self.action_dict[other_actions[k]]) % self.n for k in range(len(other_locs))]
            new_state = (agent_loc_new, other_locs_new, stay_counter)
            reward = self.get_reward(new_state)

            transitions.append((new_state, np.prod(other_probs), reward))



        # for other_action_prob in other_action_probs:
        #     other_action = other_action_prob[0]
        #     other_prob = other_action_prob[1]
        #     # if np.array_equal(other_loc, self.other.dest):
        #     #     other_loc_new = np.array(random.choice(self.starts))
        #     # else:
        #     other_loc_new = other_loc + self.action_dict[other_action]
        #     # lemcardenas: mod each coordinate or else we go off the grid
        #     other_loc_new[0] %= self.n
        #     other_loc_new[1] %= self.n
        #
        #     if np.array_equal(agent_loc_new, agent_loc):
        #         stay_counter += 1
        #     else:
        #         stay_counter = 0
        #
        #     new_state = (agent_loc_new, other_loc_new, stay_counter)
        #     reward = self.get_reward(new_state)
        #     transitions.append((new_state, other_prob, reward))

        # if self.debug:
        #     print(transitions)
        #     self.debug_count += 1
        #     if self.debug_count > 10:
        #         self.debug = False

        return transitions

    def discount(self):
        # potentially could be updated
        return 1

    def get_reward(self, state):
        agent_loc = state[0]
        other_locs = state[1]
        stay_counter = state[2]

        for other_loc in other_locs:
            if np.array_equal(agent_loc, other_loc):
                return -100

        if np.array_equal(agent_loc, self.dest):
            return 20

        if self.grid[agent_loc[0], agent_loc[1]] == 0: # if offroad
            return -10

        if tuple(agent_loc) in self.stops and stay_counter == 1: # incentivize stopping at the stop sign
            if self.stopped:
                return -1
            self.stopped = True
            return 5

        return -1 # default, pushes toward destination so it doesn't sit in one spot

class OtherActor:
    def __init__(self, start, dest, probstop=0.9, gridsz=6):
        self.probstop = probstop
        # assume dest is a coordinate i.e. (3, 0), (2, 5)
        # assume start is a coordinate i.e. (3, 0), (2, 5)
        self.dest = dest
        self.start = start
        self.valid_locs = set()
        if start[0] == 0 :
            for i in range(0, dest[0] + 1):
                self.valid_locs.add((i, start[1]))
        elif start[0] == (gridsz - 1):
            for i in range(dest[0], gridsz):
                self.valid_locs.add((i, start[1]))
        elif start[0] == (gridsz // 2):
            for i in range(0, dest[1] + 1):
                self.valid_locs.add((start[0], i))
        else: # start[0] == ((gridsz // 2) + 1);
            for i in range(dest[1], gridsz):
                self.valid_locs.add((start[0], i))
        if dest[0] == 0 :
            for i in range(0, start[0] + 1):
                self.valid_locs.add((i, dest[1]))
        elif dest[0] == (gridsz - 1):
            for i in range(start[0], gridsz):
                self.valid_locs.add((i, dest[1]))
        elif dest[0] == (gridsz // 2):
            for i in range(0, start[1] + 1):
                self.valid_locs.add((dest[0], i))
        else: # dest[0] == ((gridsz // 2) + 1);
            for i in range(start[1], gridsz):
                self.valid_locs.add((dest[0], i))
        # print(self.valid_locs)

        self.action_dict = {'north':np.array([-1,0]),
                            'south':np.array([1,0]),
                            'west':np.array([0,-1]),
                            'east':np.array([0,1]),
                            'stay':np.array([0,0])}

    def min_manhattan_dist(self, moves, dest):
        return sorted(moves, key=lambda x: abs(x[0][0] - dest[0]) + abs(x[0][1] - dest[1]))[0]

    def get_action_probs(self, curr_loc, grid, stops):
        # assume top left corner is (0, 0)
        # assume that the grid is size 6 x 6
        #west = ((((curr_loc[0] - 1) % 6), curr_loc[1]), 'west')
        #east = ((((curr_loc[0] + 1) % 6), curr_loc[1]), 'east')
        #north = ((curr_loc[0], ((curr_loc[1] - 1) % 6)), 'north')
        #south = ((curr_loc[0], ((curr_loc[1] + 1) % 6)), 'south')
        # lemcardenas: prevents illegally wrapping around board with one move
        west  = (curr_loc + self.action_dict['west'], 'west')
        east  = (curr_loc + self.action_dict['east'], 'east')
        north = (curr_loc + self.action_dict['north'], 'north')
        south = (curr_loc + self.action_dict['south'], 'south')


        stay = (curr_loc, 'stay')
        possible_moves = [move for move in [west, east, north, south, stay] if tuple(move[0]) in self.valid_locs]
        possible_move = self.min_manhattan_dist(possible_moves, self.dest)
        if tuple(curr_loc) not in stops:
            # print([(possible_move[-1], 1)]) # ncomly added [-1] to all moves to remove the location
            return [(possible_move[-1], 1)]
        else:
            # print([('stay', self.probstop)] + [(possible_move[-1], (1 - self.probstop))])
            return [('stay', self.probstop)] + [(possible_move[-1], (1 - self.probstop))]


if __name__ == '__main__':
    grid = np.zeros((6,6))
    grid[:,2] = 1
    grid[:,3] = 1
    grid[2,:] = 1
    grid[3,:] = 1
    stops = [(1,2), (3,1), (4,3), (2,4)]

    mdp = FourWayStopMDP(grid, stops, num_other=3)
    start_state = mdp.startState()
    print(start_state)
    # print(mdp.get_reward(start_state))
    # print(mdp.dest)
    # print(start_state)
    # print(mdp.succAndProbReward(start_state, 'north'))
