#! /usr/bin/env python


class Logger():

	def __init__(self, place):
		self.place = place

	def log(self, message, additional_tag='', tag='INFO'):
		print('[ {} {}]  {}.'.format(tag, additional_tag, message))