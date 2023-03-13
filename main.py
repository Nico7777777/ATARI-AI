from ale_py import ALEInterface, SDL_SUPPORT
from collections import defaultdict
from ale_py.roms import Alien
import numpy as np
import itertools
import random
import time


def greedy_policy(Q, epsilon, num_actions):
    def policy_function(state):
        #sunt 6 actiuni cu posibilitati - 2K, K, 2K, k, K, 2K
        #num_actions = 6
        K = epsilon / (num_actions + 4)
        action_probabilities = np.ones(num_actions,
                                       dtype=float) * K
        action_probabilities[2] = action_probabilities[5] = 3*K
        #print(action_probabilities)# suma lor e 1 acuma, ca inainte nu era
        #if sum(action_probabilities) > 1:
        #    action_probabilities[0] -= sum(action_probabilities) - 1
        #print(sum(action_probabilities))
        best_action = np.argmax(Q[state])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policy_function


def q_learning(env, num_episodes, discount_factor=1.7, alpha=0.75, epsilon=0.1) -> defaultdict:

    temporar_legal_actions = env.getLegalActionSet()
    #print(temporar_legal_actions)
    legal_actions = temporar_legal_actions[0:6]
    #print(legal_actions)
    num_actions = len(legal_actions)
    
    #with open("actiuni.txt", 'w') as f:
    #    for i in legal_actions:
    #        f.write(str(i) + '\n')
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Create an epsilon greedy policy function appropriately for environment action space. For every episode:
    policy = greedy_policy(Q, epsilon, num_actions)
    for _ in range(num_episodes):
        env.reset_game()

        state = hash(env.getScreen().tobytes())

        x, y = len(env.getScreen()), len(env.getScreen()[0])

        for _ in itertools.count():
            #print("state = ", state)
            action_probabilities = policy(state)
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            reward = env.act(action)
            next_state = hash(env.getScreen().tobytes())

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if env.game_over():
                break
            state = next_state
    return Q


if __name__ == "__main__":
    random.seed(time.time())
    ale = ALEInterface()
    ale.setInt("random_seed", random.randint(1, 1000000000))
    ale.setFloat("repeat_action_probability", 0.37)  # ca cine stie, se mai intampla chestii repetitive
    ale.loadROM(Alien)

    if SDL_SUPPORT:
        ale.setBool("sound", True)
        ale.setBool("display_screen", True)

    ale.loadROM(Alien)  # -- aici chiar se porneste jocul
    modes = ale.getAvailableModes()
    diffs = ale.getAvailableDifficulties()

    for mode in modes:
        for diff in diffs:
            ale.setDifficulty(diff)
            ale.setMode(mode)
            ale.reset_game()
            print(f"Mode {mode} difficulty {diff}:")
            a = q_learning(ale, 10)
            print("Q=", a)
