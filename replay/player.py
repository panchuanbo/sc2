from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

import datetime

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from train import ActorNetwork

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

_POSSIBLE_ACTIONS = [_NO_OP, _MOVE_SCREEN, _ATTACK_SCREEN, _SELECT_BOX, _SELECT]
_ACTION_MAP = np.array([[1, 0, 0, 0, 0],
[1, 1, 1, 0, 0],
[1, 1, 1, 0, 0],
[1, 1, 1, 1, 1],
[1, 1, 1, 0, 0]])

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
EP_MULTIPLIER = 0.095 #(10000 steps to 0.1)
NEGATIVE_INFINITY = -float('inf')

class ImitationAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.sess = tf.Session()
        self.actor = ActorNetwork(self.sess, ACTOR_LEARNING_RATE, TAU)
        self.sess.run(tf.global_variables_initializer())
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

        self.rewards_list = []

        saver = tf.train.Saver()
        saver.restore(self.sess, "./imitation_learner")

    def __del__(self):
        file_name = 'rewards_' + str(datetime.datetime.now()) + '.txt'
        file_name = file_name.replace(":", ".")
        rewards_file = open(file_name, 'w')
        for r in self.rewards_list:
            rewards_file.write("%s\n" % str(r))

    def reset(self):
        print("Episode " + str(self.episodes) + ": " + str(self.reward))
        self.rewards_list.append(self.reward)
        super(ImitationAgent, self).reset()
        self.reward = 0
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

    def step(self, obs):
        global EPSILON
        super(ImitationAgent, self).step(obs)
        currentstate = np.transpose(obs.observation["screen"], [1, 2, 0]) #84x84x17
        function_id, args = self.networktosc2(self.actor.predict(np.reshape(currentstate, [1, 84, 84, 17]))[0], obs)

        return actions.FunctionCall(function_id, args)

    #helper methods for converting moves between sc2 and the neural network

    def sc2tonetwork(self, function_id, args):
        id_index = _POSSIBLE_ACTIONS.index(function_id)
        args = np.ndarray.flatten(np.array(args[1:]))
        result = np.zeros((5, 5))
        result[id_index,0] = 1
        for i in range(len(args)):
            result[id_index,i + 1] = args[i]
        return result

    def networktosc2(self, network_action, obs):
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