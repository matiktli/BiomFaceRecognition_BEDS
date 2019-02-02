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
		
		self.imageX = imageSize[0]
		self.imageY = imageSize[1]
		self.imageS = self.imageX * self.imageY

		self.n_classes = 30 #num of peoples? idk now
		self.batch_size = 100

		self.x = tf.placeholder('int', [None, self.imageX])
		self.y = tf.placeholder('int')


	def neaural_netwrk_model(self, data):
		hidden_1_layer = {'wieghts': tf.Variable(tf.radom_normal([self.imageS, self.n_nodes_hl1])),
						  'biases': tf.Variable(tf.radom_normal([]))}   
						  #https://www.youtube.com/watch?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&time_continue=1093&v=BhpvH5DuVu8