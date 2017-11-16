import tensorflow as tf
import numpy as np
import os.path

LAYER_SPECS = [32, 64, 8, 128, 8]

class ActorNetwork(object):

	def __init__(self, sess, learning_rate, tau):
		self.sess = sess
		self.learning_rate = learning_rate
		self.tau = tau

		# Actor Network
		self.inputs, self.is_training, self.out = self.model()

		self.network_params = tf.trainable_variables()

		# Target Network
		self.target_inputs, self.target_is_training, self.target_out = self.model()

		self.target_network_params = tf.trainable_variables()[
			len(self.network_params):]

		# Op for periodically updating target network with online network
		# weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# This gradient will be provided by the critic network
		self.action_gradient = tf.placeholder(tf.float32, [None, 5, 5])

		# Combine the gradients here
		self.actor_gradients = tf.gradients(
			self.out, self.network_params, -self.action_gradient)

		# Optimization Op
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
			apply_gradients(zip(self.actor_gradients, self.network_params))

		self.num_trainable_vars = len(
			self.network_params) + len(self.target_network_params)

	def model(self):
		state = tf.placeholder(tf.float32, [None, 84, 84, 17])
		is_training = tf.placeholder(tf.bool)

		out = tf.layers.batch_normalization(state, training = is_training)
		out = tf.layers.conv2d(out, 32, 5, 1, "SAME")
		out = tf.nn.relu(out)
		for i in range(3):
			out = tf.layers.conv2d(out, LAYER_SPECS[i], 3, 1, "SAME")
			if i % 2 == 1:
				out = tf.layers.batch_normalization(out, training = is_training)
				out = tf.layers.max_pooling2d(out, 2, 2)
			out = tf.nn.relu(out)

		out = tf.layers.dense(tf.contrib.layers.flatten(out), 300, activation = tf.nn.relu)
		return state, is_training, tf.reshape(tf.layers.dense(out, 25), [-1, 5,5])

	def train(self, inputs, a_gradient):
		self.sess.run(self.optimize, feed_dict={
			self.inputs: inputs,
			self.is_training: True,
			self.action_gradient: a_gradient
		})

	def predict(self, inputs):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.is_training: False
		})

	def predict_target(self, inputs):
		return self.sess.run(self.target_out, feed_dict={
			self.target_inputs: inputs,
			self.target_is_training: False
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars

class CriticNetwork(object):
	def __init__(self, sess, learning_rate, tau, num_actor_vars):
		self.sess = sess
		self.learning_rate = learning_rate
		self.tau = tau

		# Create the critic network
		self.inputs, self.action, self.is_training, self.out = self.model()

		self.network_params = tf.trainable_variables()[num_actor_vars:]

		# Target Network
		self.target_inputs, self.target_action, self.target_is_training, self.target_out = self.model()

		self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

		# Op for periodically updating target network with online network
		# weights with regularization
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# Network target (y_i)
		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

		# Define loss and optimization Op
		self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
		self.optimize = tf.train.AdamOptimizer(
			self.learning_rate).minimize(self.loss)

		# Get the gradient of the net w.r.t. the action.
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		self.action_grads = tf.gradients(self.out, self.action)

	def model(self):
		state = tf.placeholder(tf.float32, [None, 84, 84, 17])
		action = tf.placeholder(tf.float32, [None, 5, 5])
		is_training = tf.placeholder(tf.bool)

		out = tf.layers.batch_normalization(state, training = is_training)
		out = tf.layers.conv2d(out, 32, 5, 1, "SAME")
		out = tf.nn.relu(out)
		for i in range(3):
			out = tf.layers.conv2d(out, LAYER_SPECS[i], 3, 1, "SAME")
			if i % 2 == 1:
				out = tf.layers.batch_normalization(out, training = is_training)
				out = tf.layers.max_pooling2d(out, 2, 2)
			out = tf.nn.relu(out)

		state_out = tf.layers.dense(tf.contrib.layers.flatten(out), 300, activation = tf.nn.relu)
		action_out = tf.layers.dense(tf.contrib.layers.flatten(action), 300, activation = tf.nn.relu)
		out = state_out + action_out
		return state, action, is_training, tf.layers.dense(out, 1)

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: inputs,
			self.action: action,
			self.is_training: True,
			self.predicted_q_value: predicted_q_value
		})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.action: action,
			self.is_training: False
		})

	def predict_target(self, inputs, action):
		return self.sess.run(self.target_out, feed_dict={
			self.target_inputs: inputs,
			self.target_action: action,
			self.target_is_training: False
		})

	def action_gradients(self, inputs, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.inputs: inputs,
			self.action: actions,
			self.is_training: False
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)