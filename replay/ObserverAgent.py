#!/usr/bin/env python

import numpy as np

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

class ObserverAgent():

	def __init__(self):
		print("Observer Initialized")
		self.frames = []
		self.actions = []

	def __del__(self):
		print("Observer Finished Execution")
		npframes = np.array(self.frames)
		npactions = np.array(self.actions)
		np.save("Features2", npframes)
		np.save("Actions2", npactions)

	def step(self, time_step, actions):
		screen = np.transpose(time_step.observation["screen"], [1, 2, 0])
		for i in range(len(actions)):
			action = actions[i]

			action_array = self.sc2tonetwork(action.function, action.arguments)

			if action_array is not None:
				self.frames.append(screen) 
				self.actions.append(action_array)


	def sc2tonetwork(self, function_id, args):
		if function_id not in _POSSIBLE_ACTIONS:
			return
		id_index = _POSSIBLE_ACTIONS.index(function_id)
		args = np.ndarray.flatten(np.array(args[1:]))
		result = np.zeros((5, 5))
		result[id_index,0] = 1
		for i in range(len(args)):
			result[id_index,i + 1] = args[i]
		return result
