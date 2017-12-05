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
		self.inputs, self.is_training, self.out, self.out_q = self.model()

		self.network_params = tf.trainable_variables()

		# Target Network
		self.target_inputs, self.target_is_training, self.target_out, self.target_q_out = self.model()

		self.target_network_params = tf.trainable_variables()[
			len(self.network_params):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

		# Define loss and optimization Op
		self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out_q)
		self.optimize = tf.train.AdamOptimizer(
			self.learning_rate).minimize(self.loss)

	def model(self):
		state = tf.placeholder(tf.float32, [None, 84, 84, 17])			    # a batch of 84x84x17 grid tensors
		is_training = tf.placeholder(tf.bool) 								# Training (for batch-norm)

		out = tf.layers.batch_normalization(state, training=is_training)	# Batch-norm the first layer
		out = tf.layers.conv2d(out, 32, 5, 1, "SAME")
		out = tf.nn.relu(out)

		# (Conv -> ReLU) -> (Conv -> BN -> POOL -> ReLU) -> (Conv -> ReLU)
		for i in range(3):
			out = tf.layers.conv2d(out, LAYER_SPECS[i], 3, 1, "SAME")
			if i % 2 == 1:
				out = tf.layers.batch_normalization(out, training = is_training)
				out = tf.layers.max_pooling2d(out, 2, 2)
			out = tf.nn.relu(out)

		out = tf.layers.dense(tf.contrib.layers.flatten(out), 300, activation = tf.nn.relu) # FCN

		action_matrix = tf.reshape(tf.layers.dense(out, 9), [-1, 3, 3]) 					# Matrix of possible actions
		
		return state, is_training, action_matrix, tf.layers.dense(out, 1)

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: inputs,
			# self.target_out: 
			self.is_training: True,
			self.predicted_q_value: predicted_q_value
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