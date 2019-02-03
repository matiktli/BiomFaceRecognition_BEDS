#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import Logger as lg


class TensorService():

	def __init__(self, imageSize):
		self.LOG = lg.Logger('tf-svc')
		self.SESSION = tf.Session()
		self.hidden_layer_n_nodes = [500, 500, 500]
		
		self.imageX = imageSize[0]
		self.imageY = imageSize[1]
		self.imageS = self.imageX * self.imageY

		self.n_classes = 30 #num of peoples? idk now
		self.batch_size = 100
		self.n_steps = 10

		self.x = tf.placeholder('float', [None, self.imageX])
		self.y = tf.placeholder('float')


	# definitin of computation graph
	# take a lot params from constructor, be aware
	def neural_network_model(self, data):
		result = [None for i in range(self.hidden_layer_n_nodes.__len__())]
		counter = 0
		for node_size in self.hidden_layer_n_nodes:
			xSize = node_size
			if counter == 0:
				xSize = self.imageS
			
			if counter == self.hidden_layer_n_nodes.__len__():
				next_node_size = xSize
			else:
				next_node_size = self.hidden_layer_n_nodes[counter+1]
				
			hidden_n_layer = {
						  'id': counter,
						  # assign weights to hidden layer
						  'weights': tf.Variable(tf.radom_normal([xSize, next_node_size])),
						  # assign random biases to layer
						  'biases': tf.Variable(tf.radom_normal([next_node_size]))}
			result[counter] = hidden_n_layer
			self.LOG.log("Created layer def no.{0}, with input: {1}, and output: {2} layer sizes. And biases from range: {2}."
				.format(hidden_n_layer['id'], xSize, next_node_size), 'layer_def', 'MODELLING')
			counter += 1

		
		all_layers = process_and_return_layers(self, data, self.hidden_layer_defs)
		output = all_layers[-1]
		return output

	def process_and_return_layers(self, data, hidden_layer_defs):
		layers = [None for i in range(self.hidden_layer_n_nodes.__len__())]
		counter = 0
		for layer_def in hidden_layer_defs:
			# adding tensor:  (input_data * wieghts) + biases
			layer_n = tf.add(tf.matmul(data, layer_def['weights']), layer_def['biases'])
			# we dont want to computes rectified linear: (max(features=layer_n, 0))
			if counter != self.hidden_layer_n_nodes.__len__():
				layer_n = tf.nn.relu(layer_n)
			layers[counter] = layer_n
			self.LOG.log("Created & added layer no.{}".format(counter), 'layer','PREPARING')
			counter += 1
		return layers


	def train_neural_network(self):
		prediction = neural_network_model(self.x)
		# 'tf.nn.softmax_cross_entropy_with_logits' is our costs function
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))

		# We want to minimise our cost, this is purpose of optymizer
		# 'AdamOptymizer()' -> 0.001(default)
		optymizer = tf.train.AdamOptymizer(learning_rate = 0.001).minimize(cost)

		# 'self.n_steps' -> number of cycles
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())

			for step in range(self.n_steps):
				step_loss = 0
				for _ in range(int(mnist.train.num_examples/self.batch_size)):
					# 'mnist.train.next_batch' just function that does everything for us, training batches
					step_x, step_y = mnist.train.next_batch(self.batch_size)
					# we are optymizing cost by x and y, we are doing this by modyfing layers weights in tensorflow
					_, c = sess.run([optymizer, cost], feed_dict = {x: step_x, y: step_y})
					step_loss += c
				percentage = int((step/self.n_steps)*100)
				self.LOG.log("Completed {}\%. On step no.{}, with loss: {}".format(percentage, step, step_loss), 'step', 'TRAINING')

			# we are asserting that both values are identical (this is not what i want i think -,- but lets see what will happen)
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y,1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			#self.LOG.log('Accuracy: {}\%.'.format(accuracy.eval({x: }) * 100), 'acc', 'RESULT')