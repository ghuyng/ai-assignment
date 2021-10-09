import numpy as np
import CoGanh as cg
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
import time
from copy import deepcopy
import random

"""## Define Game Environment and Experience Buffer """

INITIAL_BOARD = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, -1],
    [-1, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1],
])

class CoGanhEnv():
    def __init__(self, start_player=1):
        self.start_player = start_player
        self.reset()

    def reset(self):
        self.current_player = self.start_player
        self.board = deepcopy(INITIAL_BOARD)
        self.prev_board = deepcopy(INITIAL_BOARD)
        self.prev_move = None

    def _update_board(self, move):
        # move = (scr, dst) --> old and new position of a chess piece
        src, dst = move
        self.prev_board = deepcopy(self.board)
        self.board = cg.board_after_move_and_capturing(src, dst, self.board)
        self.prev_move = move

    def _is_over(self):
        return np.all(self.board >= 0) or np.all(self.board <= 0)

    def _get_reward(self):
        if np.all(self.board >= 0):
            return 1
        if np.all(self.board <= 0):
            return -1
        return 0
    
    def observe(self):
        return np.concatenate((self.board, self.prev_board), axis=0)

    def act(self, move):
        self._update_board(move)
        self.current_player = -self.current_player
        reward = self._get_reward()
        game_over = reward != 0
        return self.observe(), reward, game_over

    def get_all_possible_moves(self):
        return cg.get_all_legal_moves(self.prev_board, self.board, self.current_player)

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10, filtered = False):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape
        # env_dim = model.input_shape[1]
        inputs = np.zeros((min(len_memory, batch_size), *(env_dim)))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = get_prediction(model, state=state_t, filtered_moves=None)
            all_legal_moves = None
            if filtered:
                all_legal_moves = cg.get_all_legal_moves(current_board=state_tp1[:5], old_board=state_tp1[5:], player=1)
            Q_sa = np.nanmax(get_prediction(model, state=state_tp1, filtered_moves=all_legal_moves))
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

"""## Define Machine Learning model"""

def create_model():
    hidden_size = 100
    num_actions = 5 * 5 * 8
    model = Sequential()
    model.add(Flatten(input_shape=(10, 5,)))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))

    optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(optimizer, "mse")
    return model


move_map = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
def get_move_from_action_number(action):
    """
    0. (0, 0) -> (-1, 1)
    1. (0, 0) -> (0, 1)
    2. (0, 0) -> (1, 1)
    3. (0, 0) -> (-1, 0)
    4. (0, 0) -> (1, 0)
    5. (0, 0) -> (-1, -1)
    6. (0, 0) -> (0, -1)
    7. (0, 0) -> (1, -1)
    """
    bucket = action // 8 
    src = (bucket // 5, bucket % 5)

    order = action % 8
    dst = (src[0] + move_map[order][0], src[1] + move_map[order][1])
    return src, dst

def to_action_number(move):
    # Reverse the calculation of above function
    src, dst = move
    tmp = (dst[0] - src[0], dst[1] - src[1])
    order = move_map.index(tmp)
    bucket = src[0] * 5 + src[1]
    return bucket * 8 + order

def get_prediction(model, state, filtered_moves = None):
    q = model.predict(state.reshape(1, *state.shape))
    if not filtered_moves:
        return q[0]
    filtered_action = [to_action_number(move) for move in filtered_moves]
    adjustment_mat = np.full_like(q[0], np.nan)
    adjustment_mat[filtered_action] = 1
    return q[0] * adjustment_mat

"""## Training Process"""

def train_with_policy_punishment(model, env, exp_replay, epsilon=.1):
    epoch = 100
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        start = time.time()
        while not game_over:
            input_tm1 = input_t
            # get next action for player 1
            all_legal_moves = env.get_all_possible_moves()
            if np.random.rand() <= epsilon:
                move = random.choice(all_legal_moves)

                # apply action, get rewards and new state
                input_t, reward, game_over = env.act(move)
            else:
                q = get_prediction(model, state=input_tm1, filtered_moves=None)
                action = np.argmax(q)
                move = get_move_from_action_number(action)
                if move not in all_legal_moves:
                    reward = -2
                    game_over = True
                else:
                    input_t, reward, game_over = env.act(move)


            if reward == 1:
                win_cnt += 1

            if not game_over:
                opponent_move = random.choice(env.get_all_possible_moves())
                input_t, reward, game_over = env.act(opponent_move)

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=50)

            loss += model.train_on_batch(inputs, targets)
        duration = time.time() - start
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {} | Duration {:.4f}".format(e, loss, win_cnt, duration))

def train_with_policy_filter(model, env, exp_replay, epsilon=.1, epsilon_decay=0):
    epoch = 100
    win_cnt = 0
    for e in range(71, epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        start = time.time()
        while not game_over:
            input_tm1 = input_t
            # get next action for player 1
            all_legal_moves = env.get_all_possible_moves()
            if np.random.rand() <= epsilon:
                move = random.choice(all_legal_moves)
                action = to_action_number(move)

            else:
                q = get_prediction(model, state=input_tm1, filtered_moves=all_legal_moves)
                action = np.nanargmax(q)
                move = get_move_from_action_number(action)

            input_t, reward, game_over = env.act(move)

            if reward == 1:
                win_cnt += 1

            if not game_over:
                opponent_move = random.choice(env.get_all_possible_moves())
                input_t, reward, game_over = env.act(opponent_move)

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=50, filtered=True)

            loss += model.train_on_batch(inputs, targets)
        duration = time.time() - start
        print("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Duration {:.4f}".format(e+1, epoch, loss, win_cnt, duration))
        # print("Final board: {}".format(env.board))
        epsilon = max(epsilon * np.exp(-epsilon_decay), .1)
        if e % 10 == 0:
            model.save_weights('/content/drive/MyDrive/AI-checkpoints/cpt-{}'.format(e))
            print('Checkpoint saved at epoch {}'.format(e))

## Uncomment these line to train model ##
# env = CoGanhEnv()
# exp_replay = ExperienceReplay()
# model = create_model()

# # Train with policy punishment
# train_with_policy_punishment(model, env, exp_replay, epsilon=0.1)
# # Train with policy filter and fixed epsilon
# train_with_policy_filter(model, env, exp_replay, epsilon=0.1, epsilon_decay=0)
# # Train with policy filter and decaying epsilon
# train_with_policy_filter(model, env, exp_replay, epsilon=1, epsilon_decay=0.0025)
# # Saved train model
# model.save("trained-model.h5")
###########################################

"""## Test and Evaluation"""

def test_self_play(player_model=None, opponent_model=None, match_count=100, epsilon=.1, swap_side=True):
    win_cnt = 0
    models = {
        1: player_model,
        -1: opponent_model
    }
    def play_match(env):
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        moves_made = 0
        while not game_over:
            input_tm1 = input_t
            model = models[env.current_player]
            # get next action for player 1
            all_legal_moves = env.get_all_possible_moves()
            if model is None or np.random.rand() <= epsilon:
                move = random.choice(all_legal_moves)

            else:
                q = get_prediction(model, state=env.current_player * input_tm1, filtered_moves=all_legal_moves)
                action = np.nanargmax(q)
                move = get_move_from_action_number(action)

            input_t, reward, game_over = env.act(move)
            moves_made += 1

            if reward == 1:
                return moves_made, 1
        
        return moves_made, 0
    
    env = CoGanhEnv(start_player=1)
    epoch = match_count // 2 if swap_side else match_count
    win_cnt = 0
    print("Player goes first")
    for e in range(epoch):
        moves_made, win = play_match(env)
        win_cnt += win
        win_rate = win_cnt / (e + 1)
        print("Match {:03d}/{} | Moves made {} | Win count {} | Win rate {:.4f}".format(e, match_count-1, moves_made, win_cnt, win_rate))
    if swap_side:
        print("---------------------------------------------------")
        print("Opponent goes first")
        env = CoGanhEnv(start_player=-1)
        for e in range(epoch, match_count):
            moves_made, win = play_match(env)
            win_cnt += win
            win_rate = win_cnt / (e + 1)
            print("Match {:03d}/{} | Moves made {} | Win count {} | Win rate {:.4f}".format(e, match_count-1, moves_made, win_cnt, win_rate))

# Test pretrained model
# Load pretrained model
model_fixed_eps = tf.keras.models.load_model("model-policy-filter-fixed-eps-1.h5")
model_decay_eps = tf.keras.models.load_model("model-policy-filter-decay-eps-1.h5")

# Test fixed-epsilon model against random-based opponent
test_self_play(player_model=model_fixed_eps, opponent_model=None, match_count=1000, swap_side=True)

# Test decayed-epsilon model against random-based opponent
test_self_play(player_model=model_decay_eps, opponent_model=None, match_count=1000, swap_side=True)

# Test decayed-epsilon vs fixed-epsilon model against random-based opponent
test_self_play(player_model=model_decay_eps, opponent_model=model_fixed_eps, match_count=1000, swap_side=True)