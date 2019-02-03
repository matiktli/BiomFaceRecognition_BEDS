#! /usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
import TensorService as tenUtil

IMG_SIZE = (32,32)
IMG_NUMBER = 1500
VECTOR_SIZE = IMG_SIZE[0] * IMG_SIZE[1]
FILE_PATH = '../data/face-data-' + IMG_SIZE[0].__str__() + '-' + IMG_SIZE[1].__str__() + '.csv'

def main():
	tfServ = tenUtil.TensorService(IMG_SIZE)
	vectorsAndLables = readFileToNpArray(FILE_PATH)
	tfServ.train_neural_network(vectorsAndLables)
	pass

def readFileToNpArray(pathToFile):
	result = np.empty((IMG_NUMBER, VECTOR_SIZE + 1))
	with open(pathToFile, 'r') as fileReader:
		csvReader = csv.reader(fileReader, delimiter=',')
		rowCounter = 0
		for row in csvReader:
			result[rowCounter, 0] = int(row[0])
			for pixel in range(1, VECTOR_SIZE):
				result[rowCounter,pixel] = int(row[pixel])
			rowCounter += 1
	return result


if __name__ == '__main__':
    main()