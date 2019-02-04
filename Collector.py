#! /usr/bin/env python
import matplotlib.pyplot as plt
import time

class Collector():

	def __init__(self):
		self.x = 0
		self.width_max = 100
		self.heigh_max = 400000
		self.heigh_min = 0


	def addSingleSample(self, step, sampleFrom, sampleTo, loss):
		plt.plot([step], [loss], 'bo')

	def addAfterStepSample(self, step, loss):
		plt.plot([step], [loss], 'ro')

	def addAccuraccy(self, accuracy):
		plt.title('Accuraccy: {}'.format(accuracy))

	def showPlot(self):
		plt.show()

	def savePlot(self):
		plt.ylabel('LOSS')
		plt.xlabel('STEP NO.')
		plt.savefig('../SAVED/s{}_{}.jpg'.format(self.width_max, int(round(time.time() * 1000))))

	#additional setters
	def _set_w_max(self, w_max):
		self.width_max = w_max

	def _set_h_max(self, h_max):
		self.heigh_max = h_max
	
	def _set_h_min(self, h_min):
		self.heigh_min = h_min