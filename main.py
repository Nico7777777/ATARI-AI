from collections import defaultdict
from ale_py.roms import Alien
from ale_py import ALEInterface, SDL_SUPPORT
import numpy as np
import itertools


def create_epsilon_greedy_policy(Q, epsilon, num_actions):
    def policy_function(state):
        action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policy_function


def hashing(x):
    return hash(x.tostring())


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.6, epsilon=0.1):
    legal_actions = env.getLegalActionSet()
    num_actions = len(legal_actions)
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Create an epsilon greedy policy function appropriately for environment action space
    # For every episode
    policy = create_epsilon_greedy_policy(Q, epsilon, num_actions)
    for _ in range(num_episodes):

        env.reset_game()

        state = hashing(env.getScreen())
        print(len(env.getScreen()))
        print(len(env.getScreen()[0]))

        for _ in itertools.count():
            action_probabilities = policy(state)
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            reward = env.act(action)
            done = env.game_over()
            next_state = hashing(env.getScreen())

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break
            state = next_state
    return Q


ale = ALEInterface()
ale.setInt("random_seed", 65535)
ale.setFloat("repeat_action_probability", 0.25)
if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)
ale.loadROM(Alien)
modes = ale.getAvailableModes()
diffs = ale.getAvailableDifficulties()

leg_act = ale.getLegalActionSet()

for mode in modes:
    for diff in diffs:
        ale.setDifficulty(diff)
        ale.setMode(mode)
        ale.reset_game()
        print(f"Mode {mode} difficulty {diff}:")
        q_learning(ale, 1000)
