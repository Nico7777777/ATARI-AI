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

def is_tile(matrix, start_i, start_j):
    return (matrix[start_i][start_j] == matrix[start_i+1][start_j]) and (matrix[start_i][start_j] != matrix[start_i+2][start_j])


def is_terrain(matrix, start_i, start_j):
    return not matrix[start_i][start_j] in [132, 96]


def is_me(matrix, start_i, start_j):
    return (matrix[start_i][start_j] == 142) and (not is_tile(matrix, start_i, start_j)) and (not is_terrain(matrix, start_i, start_j))

def map_state_2(matrix, culoare=142):
    num_cols = len(matrix[0])
    my_i = None
    my_j = None
    for i in range(0, 174):
        for j in range(0, num_cols):
            if is_me(matrix, i, j):
                my_i, my_j = i, j
                break
        if my_i != None and my_j != None:
            break
    if my_i == None:
        return (0, 0, 0, 0, 0, 0)
    enemy_i = None
    enemy_j = None
    tile_i = None
    tile_j = None
    radius = 5
    while enemy_i == None and title_i == None:
        #return format:
        #(my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)
        radius += 1
        i, j = my_i - radius, my_j
        for _ in range(radius+1):
            if i > 0 and i < 174 and j > 0 and j < num_cols:
                if not is_tile(matrix, i, j) and not is_terrain(matrix, i, j):
                    return (my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)
            i += 1
            j -= 1
        for _ in range(radius+1):
            if i > 0 and i < 174 and j > 0 and j < num_cols:
                if not is_tile(matrix, i, j) and not is_terrain(matrix, i, j):
                    return (my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)
            i += 1
            j += 1
        for _ in range(radius+1):
            if i > 0 and i < 174 and j > 0 and j < num_cols:
                if not is_tile(matrix, i, j) and not is_terrain(matrix, i, j):
                    return (my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)
            i -= 1
            j += 1
        for _ in range(radius+1):
            if i > 0 and i < 174 and j > 0 and j < num_cols:
                if not is_tile(matrix, i, j) and not is_terrain(matrix, i, j):
                    return (my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)
            i -= 1
            j -= 1

def map_state(matrix, culoare=142):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    my_i = None
    my_j = None
    for i in range(0, 174):
        for j in range(0, num_cols):
            if is_me(matrix, i, j):
                my_i, my_j = i, j
                break
        if my_i != None and my_j != None:
            break
    if my_i == None:
        return (0, 0, 0, 0, 0, 0)
    enemy_i = None
    enemy_j = None
    tile_i = None
    tile_j = None
    for i in range(0, num_rows):
        if enemy_i != None and tile_i != None:
            return (my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)

        current_i = my_i-i
        if current_i > 0:
            for j in range(0, num_cols):
                #la dreapta
                current_j = my_j+j
                if current_j > 0 and current_j < num_cols:
                    i_tile = is_tile(matrix, current_i, current_j)
                    i_terrain = is_terrain(matrix, current_i, current_j)
                    if tile_i == None:
                        if i_tile:
                            tile_i, tile_j = current_i, current_j
                    if enemy_i == None:
                        if not i_tile and not i_terrain:
                            if matrix[current_i][current_j] != 142:
                                enemy_i, enemy_j = current_i, current_j
                #la stanga
                current_j = my_j-j
                if current_j > 0 and current_j < num_cols:
                    i_tile = is_tile(matrix, current_i, current_j)
                    i_terrain = is_terrain(matrix, current_i, current_j)
                    if tile_i == None:
                        if i_tile:
                            tile_i, tile_j = current_i, current_j
                    if enemy_i == None:
                        if not i_tile and not i_terrain:
                            if matrix[current_i][current_j] != 142:
                                enemy_i, enemy_j = current_i, current_j


        current_i = my_i + i
        if current_i < num_rows:
            for j in range(0, num_cols):
                # la dreapta
                current_j = my_j+j
                if current_j > 0 and current_j < num_cols:
                    i_tile = is_tile(matrix, current_i, current_j)
                    i_terrain = is_terrain(matrix, current_i, current_j)
                    if tile_i == None:
                        if i_tile:
                            tile_i, tile_j = current_i, current_j
                    if enemy_i == None:
                        if not i_tile and not i_terrain:
                            if matrix[current_i][current_j] != 142:
                                enemy_i, enemy_j = current_i, current_j
                # la stanga
                current_j = my_j-j
                if current_j > 0 and current_j < num_cols:
                    i_tile = is_tile(matrix, current_i, current_j)
                    i_terrain = is_terrain(matrix, current_i, current_j)
                    if tile_i == None:
                        if i_tile:
                            tile_i, tile_j = current_i, current_j
                    if enemy_i == None:
                        if not i_tile and not i_terrain:
                            if matrix[current_i][current_j] != 142:
                                enemy_i, enemy_j = current_i, current_j
    return (my_i, my_j, enemy_i, enemy_j, tile_i, tile_j)

def q_learning(env, num_episodes, discount_factor=1.7, alpha=0.8, epsilon=0.1) -> defaultdict:
    temporar_legal_actions = env.getLegalActionSet()
    legal_actions = temporar_legal_actions[0:6]
    num_actions = len(legal_actions)
    Q = defaultdict(lambda: np.zeros(num_actions))

    # Create an epsilon greedy policy function appropriately for environment action space. For every episode:
    policy = greedy_policy(Q, epsilon, num_actions)
    for _ in range(num_episodes):
        env.reset_game()
        
        #state = hash(env.getScreen())
        #pixels = np.where(env.getScreen()==142)
        #np.transpose(pixels)
        #me = [ [pixels[0][i], pixels[1][i]] for i in range(len(pixels[0])) if check_is_me(env.getScreen(), pixels[0][i], pixels[1][i])]
        #print(me)
        mapped_state = map_state_2(env.getScreen())
        s = hash(mapped_state)
        for _ in itertools.count():
            action_probabilities = policy(s)
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)


            reward = env.act(action)
            next_state = hash(map_state(env.getScreen()))

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[s][action]
            print(alpha * td_delta)
            Q[s][action] += alpha * td_delta

            if env.game_over():
                #ale.saveScreenPNG("E:\code\MLSA\AI\ATARI\ss-again.png")
                #with open("actiuni.txt", 'w') as f:
                #    f.write(str(env.getScreen()))
                break
            s = next_state
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
            #print(f"Mode {mode} difficulty {diff}:")
            a = q_learning(ale, 1)

