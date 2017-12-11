import numpy as np
import tensorflow as tf
import collections
import sys

LAYER_SPECS = [32, 64, 8, 128, 8]
_ACTION_MAP = np.array(
	[[1, 0, 0, 0, 0],
	[1, 1, 1, 0, 0],
	[1, 1, 1, 0, 0],
	[1, 1, 1, 1, 1],
	[1, 1, 1, 0, 0]]
)

class ActorNetwork(object):

	def __init__(self, sess, learning_rate, tau):
		self.sess = sess
		self.y = tf.placeholder(tf.float32, [None, 5, 5])
		self.learning_rate = learning_rate
		#self.tau = tau

		# Actor Network
		self.inputs, self.is_training, self.out = self.model()

		#self.loss1 = tf.losses.softmax_cross_entropy(tf.reshape(self.y[:,0], [-1, 5]), logits=tf.reshape(self.out[:,0], [-1, 5]))
		#self.loss2 = tf.reduce_sum(tf.losses.mean_squared_error(tf.multiply(tf.reshape(self.y[:,0], [-1, 1, 5]), self.out[:,1:]), tf.multiply(tf.reshape(self.y[:,0], [-1, 1, 5]), self.y[:,1:])))
		#self.mean_loss = tf.reduce_mean(tf.add(self.loss1, self.loss2)) #let me take for a sec

		self.loss1 = tf.losses.softmax_cross_entropy(self.y[:,:,0], logits=self.out[:,:,0])
		self.loss2 = tf.reduce_mean(tf.losses.mean_squared_error(self.out[:,:,1:], self.y[:,:,1:]))
		self.mean_loss = tf.reduce_mean(tf.add(self.loss1, self.loss2))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate)

		# batch normalization in tensorflow requires this extra dependency
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		global_step = tf.Variable(0, trainable=False)
		with tf.control_dependencies(extra_update_ops):
		    self.train_step = optimizer.minimize(self.mean_loss, global_step = global_step)

	def model(self):
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
		state = tf.placeholder(tf.float32, [None, 84, 84, 17])
		is_training = tf.placeholder(tf.bool)

		out = tf.layers.conv2d(out, 32, 5, 1, "SAME", kernel_regularizer=regularizer)
		out = tf.nn.relu(out)
		for i in range(3):
			out = tf.layers.conv2d(out, LAYER_SPECS[i], 3, 1, "SAME", kernel_regularizer=regularizer)
			if i % 2 == 1:
				out = tf.layers.max_pooling2d(out, 2, 2)
			out = tf.nn.relu(out)

		out = tf.layers.dense(tf.contrib.layers.flatten(out), 300, activation=tf.nn.relu, kernel_regularizer=regularizer)
		return state, is_training, tf.reshape(tf.layers.dense(out, 25), [-1, 5, 5])

	def train(self, inputs, correct):
		loss, _ = self.sess.run([self.mean_loss, self.train_step], feed_dict={
			self.inputs: inputs,
			self.y: correct,
			self.is_training: True,
		})
		print('loss =', loss)
		return loss

	def validate(self, inputs, correct):
		out = self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.is_training: False
		})

		out = np.array(out)

		total, pred_correct = correct.shape[0], 0
		for i in range(correct.shape[0]):
			cur_out, cur_correct = out[i], correct[i]
			out_func, correct_func = np.argmax(cur_out[:,0]), np.argmax(correct[:,0])
			if out_func == correct_func:
				pred_correct += 1
		print('func_accuracy =', pred_correct / total)

	def predict(self, inputs):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.is_training: False
		})

def load_data(mini = False):
	X, y = None, None

	if mini == True:
		X = np.load("Features1.npy")
		y = np.load("Actions1.npy")
	else:
		X = np.append(np.load("Features1.npy"), np.load("Features2.npy"))
		y = np.append(np.load("Actions1.npy"), np.load("Actions2.npy"))

	# Process the data
	nones = [idx for idx in range(X.shape[0]) if y[idx] is None]
	X = np.delete(X, nones, axis=0)
	y = np.delete(y, nones, axis=0)

	y = np.array([stuff for stuff in y])

	return X, y

def generate_sets(num_examples):
	indexes = np.arange(num_examples)
	np.random.shuffle(indexes)

	num_train = int(num_examples * 0.8)
	num_val = int(num_examples * 0.1)
	num_test = num_examples - num_train - num_val

	train = indexes[:num_train]
	val = indexes[num_train:num_train + num_val]
	test = indexes[num_train + num_val:]

	return train, val, test

def main():
	lr = 0.001
	if len(sys.argv) > 1:
		lr = float(sys.argv[1])

	print('-------------------------------------')
	print('using learning rate', lr)
	print('-------------------------------------')

	X, y = load_data(mini = True)

	train, val, test = generate_sets(X.shape[0])

	train_size = train.shape[0]
	sess = tf.Session()
	network = ActorNetwork(sess, lr, 0)
	batchsize = 25

	sess.run(tf.global_variables_initializer())

	losses = []

	for _ in range(50):
		for i in range(train_size // batchsize):
			loss = network.train(X[train[(i* batchsize)%train_size:((i + 1) * batchsize)%train_size]], y[train[(i* batchsize)%train_size:((i + 1) * batchsize)%train_size]])
			losses.append(loss)
		train, val, test = generate_sets(X.shape[0])

	loss_file = open('losses_' + str(lr) + '.txt', 'w')
	for r in losses:
		loss_file.write("%s\n" % str(r))

	network.validate(X[val], y[val])

	saver = tf.train.Saver()
	saver.save(sess, "./imitation_learner")

if __name__ == "__main__":
    main()
