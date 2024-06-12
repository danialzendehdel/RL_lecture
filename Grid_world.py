# Grid_world

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import os

# Initialize the grid world
# The grid world is n*n matrix
# The agent can move in four directions: up, down, left, right
# The agent can not move out of the grid world, in that cases the agent will stay in the same position
# The first and last cells of grid are the start and end points correspondingly.
# The agent can not move to the start cell again after reaching the end of episode


# initialization of the grid world
grid_size = 4
gamma = 1.0  # discount factor
# reward = 0  # reward for each step

states = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
start_state = 0
end_state = grid_size * grid_size - 1
actions = ['U', 'D', 'L', 'R']
action_prob = 0.25

path = os.path.join(os.getcwd(), 'Grid_world_images')
if not os.path.exists(path):
    os.mkdir(path)


def is_terminal(state):
    return True if state == end_state else False


def get_next_state(state, action):
    row, col = state // grid_size, state % grid_size

    if action == 'U':
        row = max(row - 1, 0)
    elif action == 'D':
        row = min(row + 1, grid_size - 1)
    elif action == 'L':
        col = max(col - 1, 0)
    elif action == 'R':
        col = min(col + 1, grid_size - 1)

    return states[row, col]


def get_reward(state, action):
    next_state = get_next_state(state, action)
    return -1 if not is_terminal(next_state) else 0


def get_value(state_values, state):
    return state_values[state // grid_size, state % grid_size]


def simulation(discount, delta=1e-4, in_place=True):
    new_state_values = np.zeros((grid_size, grid_size))
    iteration = 0

    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for state in range(1, grid_size * grid_size):
            if is_terminal(state):
                continue

            v_new = 0
            for action in actions:
                next_state = get_next_state(state, action)
                reward = get_reward(state, action)

                v_new += action_prob * (reward + discount * get_value(new_state_values, next_state))

            new_state_values[state // grid_size, state % grid_size] = v_new
            # draw_image(np.round(new_state_values, decimals=2))
            # plt.savefig(f'{path}/{iteration}_{state}.png')
            # plt.close()

        iteration += 1
        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

    return new_state_values, iteration


def policy_improvement(state_values, policy):
    stable = True

    for state in range(1, grid_size * grid_size):
        if is_terminal(state):
            continue

        old_action = policy[state]
        max_return = float('-inf')
        max_action = None

        for action in actions:
            next_state = get_next_state(state, action)
            reward = get_reward(state, action)
            expected_return = reward + gamma * get_value(state_values, next_state)

            if expected_return > max_return:
                max_return = expected_return
                max_action = action

        policy[state] = max_action
        # print(policy)

        if old_action != max_action:
            stable = False

    return policy, stable


# Create a function to visualize the state values and policy
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

        # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)


#
# state_val, itr = simulation(gamma)
#
# print(state_val)
# draw_image(np.round(state_val, decimals=2))
# plt.show()

# Initialize a random policy
policy = {state: np.random.choice(actions) for state in range(grid_size * grid_size)}
print(policy)
while True:
    # Policy evaluation
    state_values, _ = simulation(gamma)
    print("State values: ", state_values)
    # Policy improvement
    policy, stable = policy_improvement(state_values, policy)

    if stable:
        print(policy)
        break
