import util
import numpy as np
from typing import List, Callable, Tuple, Any, NewType

# taken from CS 221 HW4

class FourWayStopMDP(util.MDP):
    def __init__(self, grid: np.ndarray, stops: List[Tuple]):
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
        self.other = None

    # Return the start state.
    def startState(self) -> Tuple:
        # State comes in form (agent location (2-ndarray), other location (2-ndarray), stay counter)
        # Assumption: nxn grid where n is even
        n = self.grid.shape[0]
        north_start = (0, n/2 - 1)
        north_end = (0, n/2)
        south_start = (n-1, n/2)
        south_end = (n-1, n/2 - 1)
        west_start = (n/2, 0)
        west_end = (n/2 - 1, 0)
        east_start = (n/2 - 1, n-1)
        east_end = (n/2, n-1)

        agent_loc = south_start
        self.dest = north_end
        other_loc = east_start
        self.other = OtherActor(dest=west_end)
        stay_counter = 0

        return (agent_loc, other_loc, stay_counter)

    # Return set of actions possible from |state|.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state: Tuple) -> List[str]:
        actions = ['stay']
        agent_loc = state[0]
        if agent_loc[0] > 0:
            actions.append('north')
        if agent_loc[0] < self.grid.shape[0]-1:
            actions.append('south')
        if agent_loc[1] > 0:
            actions.append('west')
        if agent_loc[1] < self.grid.shape[1]-1:
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
        other_loc = state[1]
        stay_counter = state[2]

        if agent_loc == self.dest or agent_loc == other_loc: # if at goal or crashed
            new_state = (agent_loc, other_loc, -1) # -1 in stay counter signifies end state
            reward = self.get_reward(new_state)
            return [new_state, 1, reward]

        transitions = []
        other_action_probs = self.other(other_loc, self.grid, self.stops)
        agent_locs_new = [agent_loc + self.action_dict[action] for action in self.actions(state)]

        for agent_loc_new in agent_locs_new:
            for other_action_prob in other_action_probs:
                other_action = other_action_prob[0]
                other_prob = other_action_prob[1]
                other_loc_new = other_loc + self.action_dict[other_action]

                if agent_loc_new = agent_loc:
                    stay_counter += 1
                else:
                    stay_counter = 0

                new_state = (agent_loc_new, other_loc_new, stay_counter)
                reward = self.get_reward(new_state)
                transitions.append((new_state, other_prob, reward))

        return transitions

    def discount(self):
        # potentially could be updated
        return 1

    def get_reward(new_state):
        

if __name__ == '__main__':
