#! /usr/bin/env python
import numpy as np
from PIL import Image
import os
import csv

DATA_PATH = '../data'
IMG_SIZE = 32,32
DATA_SORTED_PATH = DATA_PATH + '/data_sorted_' + IMG_SIZE[0].__str__() + '_' + IMG_SIZE[1].__str__()
VETOR_SIZE = IMG_SIZE[0] * IMG_SIZE[1]
IMG_NUMBER = 1500
FILE_VECTOR_NAME = 'face-data'
VECTORS = [[0 for x in range(VETOR_SIZE+1)] for y in range(IMG_NUMBER)]

def main():
    fileCounter = 0
    if os.path.exists(DATA_SORTED_PATH):
        raise ValueError('DATA CORRUPTED, DROP \'' + DATA_SORTED_PATH + '\' folder.')
    else:
        os.makedirs(DATA_SORTED_PATH)
    for file in os.listdir(DATA_PATH):
        if (file.endswith('.jpg')):
            label  = file[5:7]
            img = Image.open(DATA_PATH + '/' + file).resize(IMG_SIZE)
            possibleColors = groupColorRangesIntoNGroups(img, 8)
            print(possibleColors)
            imageVector = convertImageToVector(img)
            labelName = savePhotoToNewFolder(img, label, DATA_SORTED_PATH)
            VECTORS[fileCounter][0] = labelName
            for i in range(VETOR_SIZE):
                VECTORS[fileCounter][i+1] = imageVector[i]
            #VECTORS.append(imageVector)
            fileCounter+=1
    writeVectorsToFile(VECTORS)
    print('[INFO] -> processed: ' + fileCounter.__str__() + ' files, of size: ' + IMG_SIZE.__str__())


def convertImageToVector(img):
    return np.array(img).flatten()

def writeVectorsToFile(allVectors):
    with open(DATA_PATH + '/' + FILE_VECTOR_NAME + '-' + IMG_SIZE[0].__str__() + '-' + IMG_SIZE[1].__str__() + '.csv', 'w') as fileWriter:
        writer = csv.writer(fileWriter)
        writer.writerows(allVectors)

def savePhotoToNewFolder(img, label, dirToSave):
    pathToCreate = dirToSave + '/' + label
    if not os.path.exists(pathToCreate):
        os.makedirs(pathToCreate)
    num = os.listdir(pathToCreate).__len__() + 1
    if num < 10:
        num = '0' + num.__str__()
    newName = label + '_' + num.__str__()
    img.save(dirToSave + '/' + label + '/' + newName + '.jpg', 'JPEG')
    return newName

def reduceImageColorsToTwo(image):
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixelVal = pixels[i,j]
            idx = int(255/2)
            if pixelVal >= idx:
                pixelVal = 255
            else:
                pixelVal = 0
            pixels[i,j] = pixelVal

def groupColorRangesIntoNGroups(image, shape):
    dx = int(255/shape)
    possibleColors = [u*dx for u in range(shape)]
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixelVal = pixels[i,j]
            tmpVal = pixelVal
            for p in range(shape):
                if pixelVal >= p*dx and pixelVal <= (p+1)*dx:
                    tmpVal = p*dx
            pixels[i,j] = pixelVal
    return possibleColors

def findFirstFreeIdxInArray(coll):
    for z in range(coll.__len__()):
        if coll[z] == -1:
            return z
    return 0

if __name__ == '__main__':
    main()
