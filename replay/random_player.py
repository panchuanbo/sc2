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

class RandomAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.rewards_list = []

    def __del__(self):
        file_name = 'random_rewards_' + str(datetime.datetime.now()) + '.txt'
        file_name = file_name.replace(":", ".")
        rewards_file = open(file_name, 'w')
        for r in self.rewards_list:
            rewards_file.write("%s\n" % str(r))
        

    def reset(self):
        print("Episode " + str(self.episodes) + ": " + str(self.reward))
        self.rewards_list.append(self.reward)
        super(RandomAgent, self).reset()
        self.reward = 0

    def step(self, obs):
        global EPSILON
        super(RandomAgent, self).step(obs)


        allowable_actions = np.intersect1d(_POSSIBLE_ACTIONS, obs.observation["available_actions"])
        function_id = np.random.choice(allowable_actions)
        args = [[np.random.randint(0, size // 2 * 2) for size in arg.sizes] for arg in self.action_spec.functions[function_id].args]

        return actions.FunctionCall(function_id, args)