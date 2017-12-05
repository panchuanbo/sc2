from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

import datetime
import os

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from utils.deepq_network import DeepQNetwork
from utils.replay import ReplayBuffer

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3 # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_BOX = actions.FUNCTIONS.select_rect.id
_SELECT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

_POSSIBLE_ACTIONS = [_NO_OP, _MOVE_SCREEN, _ATTACK_SCREEN]
_ACTION_MAP = np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]])

EPSILON = 1
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 25
MIN_FRAMES = 1000
MAX_EPISODES = 50000
MAX_EP_STEPS = 1000
GAMMA = 0.99
EP_MULTIPLIER = 0.99
NEGATIVE_INFINITY = -float('inf')
NUM_ACTIONS = 9

class DeepQAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        ###

        self.train_during_step = True # Train during self.step() vs self.reset()
        self.use_location_heuristic = False # Instead of manually find where the agent last was, use the last action location as a heuristic
        self.recompute_action_from_location = False # doesn't use self.lastnetworklocation if true
        self.penalize_useless_actions = True # -1 if reward is 0
        self.use_reward_diff = True # cur_reward - last_reward to get reward of an action v. jst using last_reward
        self.use_deepmind_atari = False # Use deepmind alti network

        ###

        self.sess = tf.Session()
        self.network = DeepQNetwork(self.sess, ACTOR_LEARNING_RATE, TAU, self.use_deepmind_atari)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.sess.run(tf.global_variables_initializer())
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

        # maybe 'lastlocation' is a misnomer; been changing around wayy to much though
        # Might be current rn?
        self.lastlocation = None

        # Ya, instead of converting sc2-->network, I just save some stuff now
        # B/c it's annoying to know which of the actions I took
        self.lastnetworkaction = None

        self.rewards_list = []

        if os.path.isfile("./models/deepq_kite"):
            print('Attempting to Load Saved Parameters')
            saver = tf.train.Saver()
            # saver.restore(self.sess, "./models/deepq_kite")

    def __del__(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "./models/deepq_kite")

        file_name = 'rewards_' + str(datetime.datetime.now()) + '.txt'
        file_name = file_name.replace(":", ".")
        rewards_file = open(file_name, 'w')
        for r in self.rewards_list:
            rewards_file.write("%s\n" % str(r))
        

    def reset(self):
        print("Episode " + str(self.episodes) + ": " + str(self.reward) + ", Epsilon: " + str(EPSILON))
        self.rewards_list.append(self.reward)
        super(DeepQAgent, self).reset()
        self.reward = 0
        self.laststate = None
        self.lastaction = None
        self.lastreward = None
        self.lastlocation = None
        self.lastnetworkaction = None

        if (self.steps >= MAX_EP_STEPS) and self.train_during_step == False:
            for i in range(500):
                self.train_network()

            # Update target networks
            self.network.update_target_network()

    def step(self, obs):
        global EPSILON
        super(DeepQAgent, self).step(obs)
        currentstate = np.transpose(obs.observation["screen"], [1, 2, 0]) #84x84x17
        # for func_id in (_NO_OP, _MOVE_SCREEN, _ATTACK_SCREEN, _SELECT_BOX, _SELECT):
        #     print(self.action_spec.functions[func_id].args)
        # print("1", self.steps)
        if not self.laststate is None:
            network_action = self.sc2tonetwork(self.lastaction[0], self.lastaction[1]) if self.recompute_action_from_location else self.lastnetworkaction
            action_reward = (self.reward - self.lastreward) if self.use_reward_diff == True else self.lastreward
            # print('action_reward =', action_reward)
            offset = -1 if self.penalize_useless_actions and action_reward == 0 else self.lastreward
            self.buffer.add(
                self.laststate,          # S
                network_action,          # A
                action_reward,           # R
                currentstate)            # S'
            # print(network_action)
        if self.use_location_heuristic:
            if not self.laststate is None:
                self.lastlocation = self.lastlocation if self.lastaction[0] == _NO_OP else tuple(self.lastaction[1][1])
            else:
                self.get_location(currentstate)
        else:
            self.get_location(currentstate)

        if (self.steps >= MAX_EP_STEPS):
            if (EPSILON > 0.1):
                EPSILON *= EP_MULTIPLIER

            if self.train_during_step:
                self.train_network()

            # Update target networks
            self.network.update_target_network()
        function_id = None
        args = None
        # print("2", self.steps)
        move_type = 'Random!'
        if (random.random() < EPSILON):
            act = self.gen_random_action()
            one_hot = [0 for _ in range(NUM_ACTIONS)]
            one_hot[act] = 1
            function_id, args = self.networktosc2(one_hot)
            self.lastnetworkaction = np.array(one_hot)
        else:
            # print("2.5", self.steps)
            net_output = self.network.predict(np.reshape(currentstate, [1, 84, 84, 17]))[0]
            function_id, args = self.networktosc2(net_output)
            move_type = 'Network'
            self.lastnetworkaction = net_output
        
        # print('[', move_type, '] Predicted Function:', self.get_name(function_id), '\t Predicted Arguments:', args, 'Cur Reward:', self.reward)
        # print("3", self.steps)
        self.laststate = currentstate
        self.lastaction = function_id, args
        self.lastreward = self.reward
        # print('Reward =', self.reward)
        # print("4", self.steps)
        debug = False
        if debug: return actions.FunctionCall(0, []) #actions.FunctionCall(331, [[0], [1, 10]])
        # print('Function:', function_id, '\tArgs:', args)
        return actions.FunctionCall(function_id, args)


    def train_network(self):
        s_batch, a_batch, r_batch, s2_batch = self.buffer.sample_batch(MINIBATCH_SIZE)

        next_q_values = self.network.predict_target(s2_batch)
        # print('next_q_values:', next_q_values)
        next_q_values = np.max(next_q_values, axis=1)

        y_i = []
        for i in range(MINIBATCH_SIZE):
            y_i.append(r_batch[i] + GAMMA * next_q_values[i])

        # Update the network
        _, _ = self.network.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))


    #helper methods for converting moves between sc2 and the neural network

    '''
    For moves, arg mode [[0], [x, y]] where (0, 0) is top left
    '''

    '''
    Random Idea: Extreme Discretization Of the Network (?)
        - Consider only the 4 cardinal directions (N, S, E, W)
        - Consider this for only attack and not-attack O_O
    '''
    # ^ OKAY - THIS MIGHT BE COOL IF WE COULD, YOU KNOW, ACTUALLY FIND LOCATIONS CORRECTLY
    '''
    Okay, extension idea:
        - Start the 'click' at a well-known location
        - And have all adjustments be relative there-of
    '''

    # One-Hot Encoding
    # [No-Op, M. Up, M. Left, M. Down, M. Right, A. Up, A. Left, A, Down, A, Right]

    def get_location(self, currentstate):
        # Attempt to find the current location
        layer = currentstate[:,:,4]
        '''
        for r in range(84):
            dat = [str(layer[r,c]) for c in range(84)]
            print(','.join(dat))
        print('---------------------')
        '''
        rows, cols, _ = currentstate.shape
        for y in range(rows):
            for x in range(cols):
                if layer[y,x] == 1:
                    self.lastlocation = (x+1, y+1)
                    return

        # print('First Loaction =', self.lastlocation)

    def get_name(self, function_id):
        if function_id == _NO_OP: return 'NO-OP'
        elif function_id == _MOVE_SCREEN: return 'MOVE'
        elif function_id == _ATTACK_SCREEN: return 'ATTACK'

    def gen_random_action(self):
        while True:
            act = np.random.randint(0, high=9)
            if act == 0: return act
            idx = (act - 1) % 4 + 1
            if idx == 1: # Up
                new_loc = self.tuple_add(self.lastlocation, (0, -5))
            if idx == 2: # Left
                new_loc = self.tuple_add(self.lastlocation, (-5, 0))
            if idx == 3: # Down
                new_loc = self.tuple_add(self.lastlocation, (0, 5))
            if idx == 4: # Right
                new_loc = self.tuple_add(self.lastlocation, (5, 0))
            if new_loc != self.lastlocation:
                return act

    def tuple_diff(self, a, b):
        return (b[0] - a[0], b[1] - a[1])

    def tuple_add(self, a, b):
        return [min(max(a[0] + b[0], 0), 83), min(max(a[1] + b[1], 0), 83)]

    # time for some kinda hacky stuff
    #'''
    def sc2tonetwork(self, function_id, args):
        actions = [0 for _ in range(NUM_ACTIONS)]
        if function_id == _NO_OP: 
            actions[0] = 1
        else:
            newlocation = (args[1][0], args[1][1])
            offset = self.tuple_diff(self.lastlocation, newlocation)
            one_hot_location = 1 if function_id == _MOVE_SCREEN else 5

            if offset[1] < 0 or (offset == (0, 0) and self.lastlocation[1] == 0):   # up
                one_hot_location += 0
            elif offset[0] < 0 or (offset == (0, 0) and self.lastlocation[0] == 0): # left
                one_hot_location += 1
            elif offset[1] > 0 or (offset == (0, 0) and self.lastlocation[1] == 83): # down
                one_hot_location += 2
            elif offset[0] > 0 or (offset == (0, 0) and self.lastlocation[0] == 83): # right
                one_hot_location += 3
            # if offset == (0, 0): print('used location:', self.lastlocation)
            actions[one_hot_location] = 1
        return actions

    def networktosc2(self, network_action):
        idx = np.argmax(np.array(network_action))
        if idx == 0:
            return _NO_OP, []
        else:
            # Enter the wall of if-statements
            if idx == 1: # M. Up
                return _MOVE_SCREEN, [[0], self.tuple_add(self.lastlocation, (0, -5))]
            if idx == 2: # M. Left
                return _MOVE_SCREEN, [[0], self.tuple_add(self.lastlocation, (-5, 0))]
            if idx == 3: # M. Down
                return _MOVE_SCREEN, [[0], self.tuple_add(self.lastlocation, (0, 5))]
            if idx == 4: # M. Right
                return _MOVE_SCREEN, [[0], self.tuple_add(self.lastlocation, (5, 0))]
            if idx == 5: # A. Up
                return _ATTACK_SCREEN, [[0], self.tuple_add(self.lastlocation, (0, -5))]
            if idx == 6: # A. Left
                return _ATTACK_SCREEN, [[0], self.tuple_add(self.lastlocation, (-5, 0))]
            if idx == 7: # A. Down
                return _ATTACK_SCREEN, [[0], self.tuple_add(self.lastlocation, (0, 5))]
            if idx == 8: # A. Right
                return _ATTACK_SCREEN, [[0], self.tuple_add(self.lastlocation, (5, 0))]

    #'''

    '''
    def sc2tonetwork(self, function_id, args):
        id_index = _POSSIBLE_ACTIONS.index(function_id)
        args = np.ndarray.flatten(np.array(args[1:]))
        result = np.zeros((3, 3))
        result[id_index,0] = 1
        for i in range(len(args)):
            result[id_index,i + 1] = args[i]
        return result

    def networktosc2(self, network_action, obs):
        print(network_action)
        index = np.argmax(network_action, axis=0)[0]
        while not _POSSIBLE_ACTIONS[index] in obs.observation["available_actions"]:
            network_action[index,0] = NEGATIVE_INFINITY
            index = np.argmax(network_action, axis=0)[0]
        function_id = _POSSIBLE_ACTIONS[index]
        args = []
        thingargs = self.action_spec.functions[function_id].args
        i = 1
        # print("2.75", self.steps)
        if index > 0: # not noop
            args = [[0]]
            for arg in self.action_spec.functions[function_id].args[1:]:
                miniargs = []
                for size in arg.sizes:
                    miniargs.append(int(max(0, min(size - 1, network_action[index, i]))))
                    i += 1
                args.append(miniargs)
        return function_id, args
        '''