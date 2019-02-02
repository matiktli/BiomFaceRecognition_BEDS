#! /usr/bin/env python
import tensorflow as tf
import numpy as np


#https://pythonprogramming.net/tensorflow-deep-neural-network-machine-learning-tutorial/?completed=/tensorflow-introduction-machine-learning-tutorial/
class TensorService():

	def __init__(self):
		self.SESSION = tf.Session()
