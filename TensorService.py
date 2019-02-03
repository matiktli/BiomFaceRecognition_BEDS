#! /usr/bin/env python
import tensorflow as tf
import numpy as np


#https://pythonprogramming.net/tensorflow-deep-neural-network-machine-learning-tutorial/?completed=/tensorflow-introduction-machine-learning-tutorial/
class TensorService():

	def __init__(self, imageSize):
		self.SESSION = tf.Session()
		self.n_nodes_hl1 = 500
		self.n_nodes_hl2 = 500
		self.n_nodes_hl3 = 500
		self.hidden_layer_n_nodes = [500, 500, 500]
		
		self.imageX = imageSize[0]
		self.imageY = imageSize[1]
		self.imageS = self.imageX * self.imageY

		self.n_classes = 30 #num of peoples? idk now
		self.batch_size = 100

		self.x = tf.placeholder('float', [None, self.imageX])
		self.y = tf.placeholder('float')


	#definitin of computation graph
	# take a lot params from constructor, be aware
	# input_size, output_size = self.imageS
	# hidden_layer_n_nodes = self.hidden_layer_n_nodes
	def neural_network_model_parametraized(self, data):
		result = [None for i in range(self.hidden_layer_n_nodes.__len__())]
		counter = 0
		for node_size in self.hidden_layer_n_nodes:
			xSize = node_size
			if counter == 0:
				xSize = self.imageS
			
			if counter == self.hidden_layer_n_nodes.__len__():
				next_node_size = xSize
				biases = tf.Variable(tf.radom_normal([next_node_size]))
			else:
				next_node_size = self.hidden_layer_n_nodes[counter+1]
				biases = tf.Variable(tf.radom_normal(next_node_size))
				
			hidden_n_layer = {
						  'id': counter,
						  # assign weights to hidden layer
						  'weights': tf.Variable(tf.radom_normal([xSize, next_node_size])),
						  # assign random biases to layer
						  'biases': biases}
			result[counter] = hidden_n_layer
			print("Created layer def no.{0}, with input: {1}, and output: {2} layer sizes. And biases from range: {2}."
				.format(hidden_n_layer['id'], xSize, next_node_size))
			counter += 1

		
		all_layers = process_and_return_layers(self, data, self.hidden_layer_defs)
		output = all_layers[-1]
		print("Output layer ready. {}", output) 

	def process_and_return_layers(self, data, hidden_layer_defs):
		layers = [None for i in range(self.hidden_layer_n_nodes.__len__())]
		counter = 0
		for layer_def in hidden_layer_defs:
			# adding tensor:  (input_data * wieghts) + biases
			layer_n = tf.add(tf.matmul(data, layer_def['weights']) + layer_def['biases'])
			# we dont want to computes rectified linear: (max(features=layer_n, 0))
			if counter != self.hidden_layer_n_nodes.__len__():
				layer_n = tf.nn.relu(layer_n)
			layers[counter] = layer_n
			print("Created & added layer no.{}".format(counter))
			counter += 1
		return layers
