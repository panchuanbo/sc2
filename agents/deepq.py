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

from utils.network import ActorNetwork
from utils.network import CriticNetwork
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
EP_MULTIPLIER = 0.99999 #(10000 steps to 0.1)
NEGATIVE_INFINITY = -float('inf')

class DeepQAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.sess = tf.Session()
        self.actor = ActorNetwork(self.sess, ACTOR_LEARNING_RATE, TAU)
        self.critic = CriticNetwork(self.sess, CRITIC_LEARNING_RATE,
            TAU, self.actor.get_num_trainable_vars())
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.sess.run(tf.global_variables_initializer())
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

        self.rewards_list = []

        saver = tf.train.Saver()
        saver.restore(self.sess, "./models/actorcritic_lingbane")

    def __del__(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "./models/actorcritic_lingbane")

        file_name = 'rewards_' + str(datetime.datetime.now()) + '.txt'
        file_name = file_name.replace(":", ".")
        rewards_file = open(file_name, 'w')
        for r in self.rewards_list:
            rewards_file.write("%s\n" % str(r))
        

    def reset(self):
        print("Episode " + str(self.episodes) + ": " + str(self.reward))
        self.rewards_list.append(self.reward)
        super(DeepQAgent, self).reset()
        self.reward = 0
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

    def step(self, obs):
        global EPSILON
        super(DeepQAgent, self).step(obs)
        currentstate = np.transpose(obs.observation["screen"], [1, 2, 0]) #84x84x17
        # for func_id in (_NO_OP, _MOVE_SCREEN, _ATTACK_SCREEN, _SELECT_BOX, _SELECT):
        #     print(self.action_spec.functions[func_id].args)
        # print("1", self.steps)
        if not self.laststate is None:
            self.buffer.add(self.laststate,
                self.sc2tonetwork(self.lastaction[0], self.lastaction[1]),
                self.lastreward, currentstate)
        if (self.steps >= MAX_EP_STEPS):
            s_batch, a_batch, r_batch, s2_batch = \
                    self.buffer.sample_batch(MINIBATCH_SIZE)

            target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for i in range(MINIBATCH_SIZE):
                y_i.append(r_batch[i] + GAMMA * target_q[i])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()
        function_id = None
        args = None
        # print("2", self.steps)
        if (random.random() < EPSILON):
            allowable_actions = np.intersect1d(_POSSIBLE_ACTIONS,
                obs.observation["available_actions"])
            function_id = np.random.choice(allowable_actions)
            args = [[np.random.randint(0, size // 2 * 2) for size in arg.sizes]
                        for arg in self.action_spec.functions[function_id].args]
        else:
            # print("2.5", self.steps)
            function_id, args = self.networktosc2(self.actor.predict(
                np.reshape(currentstate, [1, 84, 84, 17]))[0], obs)
        # print("3", self.steps)
        self.laststate = currentstate
        self.lastaction = function_id, args
        self.lastreward = self.reward
        if (EPSILON > 0.1):
            EPSILON *= EP_MULTIPLIER
        # print("4", self.steps)
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