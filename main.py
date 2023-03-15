from ale_py import ALEInterface, SDL_SUPPORT
from collections import defaultdict
from ale_py.roms import Alien
import numpy as np
import itertools
import random
import time
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

valori_pixeli = [0, 30, 78, 94, 96, 132, 142, 190]
'''
steluta -> (30, 78, 94, 142)
monstrulet -> (30, 94, 190) PRESPUNAND CA NU SE SCHIMBA SI CULORILE MONSTRIILOR ALE NAIBII


0 - e din afara tablei de joc
30 - steluta/monstrulet
78 - steluta/plusul de sus
94 - steluta
94 - steluta
96 - e movul
132 - e culoarul de mers
142 - EU/steluta/e culoarea cu care se tine scorul
190 - monstrulet
'''
def greedy_policy(Q, epsilon, num_actions):
    def policy_function(state):
        #sunt 6 actiuni cu posibilitati - 2K, K, 2K, k, K, 2K
        #num_actions = 6
        K = epsilon / (num_actions + 4)
        action_probabilities = np.ones(num_actions,
                                       dtype=float) * K
        action_probabilities[2] = action_probabilities[5] = 3*K
        #print(sum(action_probabilities))
        best_action = np.argmax(Q[state])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policy_function

def check_is_me(sc:np.ndarray, x:int, y:int)->bool:
    if sc[x-1][y] != 142 and sc[x+1][y] != 142 and sc[x][y-1] != 142 and sc[x][y+1] != 142:
        return False
    return True


def q_learning(env, num_episodes, discount_factor=1.7, alpha=0.75, epsilon=0.1) -> defaultdict:
    temporar_legal_actions = env.getLegalActionSet()
    legal_actions = temporar_legal_actions[0:6]
    num_actions = len(legal_actions)
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Create an epsilon greedy policy function appropriately for environment action space. For every episode:
    policy = greedy_policy(Q, epsilon, num_actions)
    for _ in range(num_episodes):
        env.reset_game()
        
        state = hash(env.getScreen().tobytes())
        pixels = np.where(env.getScreen()==142)
        np.transpose(pixels)
        me = [ [pixels[0][i], pixels[1][i]] for i in range(len(pixels[0])) if check_is_me(env.getScreen(), pixels[0][i], pixels[1][i])]
        print(me)
        for _ in itertools.count():
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
                #ale.saveScreenPNG("E:\code\MLSA\AI\ATARI\ss-again.png")
                #with open("actiuni.txt", 'w') as f:
                #    f.write(str(env.getScreen()))
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

    for mode in modes[0:1]:
        for diff in diffs[0:1]:
            ale.setDifficulty(diff)
            ale.setMode(mode)
            ale.reset_game()
            #print(f"Mode {mode} difficulty {diff}:")
            a = q_learning(ale, 1)

