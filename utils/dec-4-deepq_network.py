import tensorflow as tf
import numpy as np
import os.path

LAYER_SPECS = [32, 64, 8, 128, 8] 

class DeepQNetwork(object):
	def __init__(self, sess, learning_rate, tau):
		self.sess = sess
		self.learning_rate = learning_rate
		self.tau = tau

		# Actor Network
		self.inputs, self.is_training, self.out = self.model()

		# Parameters of the Actor Network
		self.network_params = tf.trainable_variables()

		# Target Network
		self.target_inputs, self.target_is_training, self.target_out = self.model()

		# Parameters of the Target Network
		self.target_network_params = tf.trainable_variables()[len(self.network_params):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# Predicted Q-Values - Should be Reward + (Discount) * max_a Q(s', a')
		self.target_q_value = tf.placeholder(tf.float32, [None, 1])

		# Actions - Should be a batch of one-hot
		self.actions = tf.placeholder(tf.float32, [None, 9])

		# Define loss and optimization Op
		self.loss = tf.losses.mean_squared_error(self.target_q_value, tf.reduce_max(tf.multiply(self.out, self.actions), axis=1, keep_dims=True))
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	def model(self):
		state = tf.placeholder(tf.float32, [None, 84, 84, 17])
		is_training = tf.placeholder(tf.bool)

		use_deepmind_atari = False

		out = tf.layers.batch_normalization(state, training=is_training)
		if use_deepmind_atari:
			# DeepMind's Atari Network (for fun)
			out = tf.layers.conv2d(out, 32, 8, 4)
			out = tf.layers.conv2d(out, 64, 4, 2)
			out = tf.layers.conv2d(out, 64, 3, 1)
			out = tf.layers.dense(tf.contrib.layers.flatten(out), 300, activation=tf.nn.relu)
		else:
			out = tf.layers.conv2d(out, 32, 5, 1, "SAME")
			out = tf.nn.relu(out)
			for i in range(3):
				out = tf.layers.conv2d(out, LAYER_SPECS[i], 3, 1, "SAME")
				if i % 2 == 1:
					out = tf.layers.batch_normalization(out, training = is_training)
					out = tf.layers.max_pooling2d(out, 2, 2)
				out = tf.nn.relu(out)

			out = tf.layers.dense(tf.contrib.layers.flatten(out), 300, activation = tf.nn.relu)


		return state, is_training, tf.layers.dense(out, 9)

	def train(self, inputs, actions, target_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: inputs,
			self.actions: actions,
			self.is_training: True,
			self.target_q_value: target_q_value
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