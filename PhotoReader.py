#! /usr/bin/env python
import numpy as np
from PIL import Image
import os
import csv

DATA_PATH = '../data'
IMG_SIZE = 32,32
IMG_NUMBER = 1500
FILE_VECTOR_NAME = 'face-data'
VECTORS = []

def main():
    fileCounter = 0
    for file in os.listdir(DATA_PATH):
        if (file.endswith('.jpg')):
            img = Image.open(DATA_PATH + '/' + file).resize(IMG_SIZE)
            imageVector = convertImageToVector(img)
            VECTORS.append(imageVector)
            fileCounter+=1
    writeVectorsToFile(VECTORS)
    print('[INFO] -> processed: ' + fileCounter.__str__() + ' files, of size: ' + IMG_SIZE.__str__())


def convertImageToVector(img):
    return np.array(img).flatten()

def writeVectorsToFile(allVectors):
    with open(DATA_PATH + '/' + FILE_VECTOR_NAME + '-' + IMG_SIZE[0].__str__() + '-' + IMG_SIZE[1].__str__() + '.csv', 'w') as fileWriter:
        writer = csv.writer(fileWriter)
        writer.writerows(allVectors)

if __name__ == '__main__':
    main()
