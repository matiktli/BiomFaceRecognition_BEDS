#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import Logger as lg


class TensorService():

	def __init__(self, imageSize):
		self.LOG = lg.Logger('tf-svc')
		self.SESSION = tf.Session()
		self.hidden_layer_n_nodes = [500,500]
		
		self.imageX = imageSize[0]
		self.imageY = imageSize[1]
		self.imageS = int(self.imageX * self.imageY)

		self.n_classes = 30 #num of people? idk now
		self.batch_size = 90
		self.n_steps = 30

		self.x = tf.placeholder('float', [None, self.imageS])
		self.y = tf.placeholder('float')


	# definitin of computation graph
	# take a lot params from constructor, be aware
	def neural_network_model(self, data):
		result = [None for i in range(self.hidden_layer_n_nodes.__len__()+1)]
		counter = 0
		for node_size in self.hidden_layer_n_nodes:
			if counter == 0:
				val_x = self.imageS
			else:
				val_x = self.hidden_layer_n_nodes[counter-1]

			hidden_n_layer = {
						  'id': counter,
						  # assign weights to hidden layer
						  'weights': tf.Variable(tf.random_normal([val_x, node_size])),
						  # assign random biases to layer
						  'biases': tf.Variable(tf.random_normal([node_size]))}
			result[counter] = hidden_n_layer
			self.LOG.log("Created layer def no.{0}, with input: {1}, and output: {2} layer sizes. And biases from range: {2}."
				.format(hidden_n_layer['id'], val_x, node_size), 'layer_def', 'MODELLING')
			counter += 1

		output_layer = {
						  'id': counter,
						  # assign weights to hidden layer
						  'weights': tf.Variable(tf.random_normal([self.hidden_layer_n_nodes[counter-1], self.n_classes])),
						  # assign random biases to layer
						  'biases': tf.Variable(tf.random_normal([self.n_classes]))}
		result[counter] = output_layer
		self.LOG.log("Created OUTPUT layer def no.{0}, with input: {1}, and output: {2} layer sizes. And biases from range: {2}."
				.format(output_layer['id'], self.hidden_layer_n_nodes[counter-1], self.n_classes), 'layer_def', 'MODELLING')

		
		all_layers = self.process_and_return_layers(data, result)
		output = all_layers[-1]
		return output

	def process_and_return_layers(self, data, hidden_layer_defs):
		layers = [None for i in range(hidden_layer_defs.__len__())]
		counter = 0
		for layer_def in hidden_layer_defs:
			if counter == 0:
				data_to_process = data  #is this right ?????
			else:
				data_to_process = layer_n
			# adding tensor:  (input_data * wieghts) + biases
			layer_n = tf.add(tf.matmul(data_to_process, layer_def['weights']), layer_def['biases'])
			# we dont want to computes rectified linear: (max(features=layer_n, 0))
			if counter != self.hidden_layer_n_nodes.__len__():
				layer_n = tf.nn.relu(layer_n)
			layers[counter] = layer_n
			self.LOG.log("Created & added layer no.{}".format(counter), 'layer','PREPARING')
			counter += 1
		return layers


	def train_neural_network(self, all_data_input):
		train_x, train_y, test_x, test_y = self.prepare_data_for_algorithm(all_data_input)		
		prediction = self.neural_network_model(self.x)
		# 'tf.nn.softmax_cross_entropy_with_logits' is our costs function
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = self.y))

		# We want to minimise our cost, this is purpose of optymizer
		# 'AdamOptimizer()' -> 0.001(default)
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

		# 'self.n_steps' -> number of cycles
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())

			for step in range(self.n_steps):
				step_loss = 0

				i = 0
				while i < len(train_x):
					start = i
					end = i+self.batch_size

					batch_x = np.array(train_x[start:end])
					batch_y = np.array(train_y[start:end])
					

					# we are optimizing cost by x and y, we are doing this by modyfing layers weights in tensorflow
					_, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
					step_loss += c
					i += self.batch_size
				percentage = int((step/self.n_steps)*100)
				self.LOG.log("Completed {}%. On step no.{}, with loss: {}".format(percentage, step, step_loss), 'step', 'TRAINING')

			# we are asserting that both values are identical (this is not what i want i think -,- but lets see what will happen)
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			self.LOG.log('Accuracy: {}%.'.format(accuracy.eval({self.x: test_x, self.y: test_y}) * 100), 'acc', 'RESULT')

	# 0.8 -> 80% of data is prepared for training and 20% for testing
	def prepare_data_for_algorithm(self, all_data_input, proportion=0.8):
		#np.random.shuffle(all_data_input)
		idx = int(all_data_input.__len__()*proportion)
		train_x_y = all_data_input[:idx, :]
		test_x_y = all_data_input[idx:,:]
		train_x, train_y = self.extract_labels_from_data(train_x_y)
		test_x, test_y = self.extract_labels_from_data(train_x_y)
		# change into vectors one hot
		train_y = self.vectorize_labels(train_y)
		test_y = self.vectorize_labels(test_y)

		self.LOG.log('Prepared data. Train: {} records, Test: {} records.'.format(train_x.__len__(), test_x.__len__()), 'form', 'PREP DATA')
		return train_x, train_y, test_x, test_y

	def extract_labels_from_data(self, data):
		data_x = data[:, [i for i in range(1, data[0].__len__())]] # 1 -> since first column is label
		data_y = data[:,0]
		return data_x, data_y

	def vectorize_labels(self, labels):
		res = [[None for _ in range(self.n_classes)] for _ in range(labels.__len__())]
		for i in range(labels.__len__()):
			res[i] = self.change_single_label_to_vector(labels[i])
		return res

	def change_single_label_to_vector(self, label):
		vect = [None for _ in range(self.n_classes)]
		for i in range(self.n_classes):
			if i == int(label):
				vect[i] = 1
			else:
				vect[i] = 0
		return vect



