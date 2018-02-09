import numpy as np
import os
import cv2
import random


def Load(path):
    labels = []
    values = []

    files = os.listdir(path)
    random.shuffle(files)
    labels = np.zeros((1,len(files)))

    index = 0
    
    for filename in files:
        in_name = path + '/' + filename
        img = cv2.imread(in_name, cv2.IMREAD_COLOR)
        
        flat = np.reshape(img, np.prod(img.shape)) / 255
        label = 1 if "yes" in filename else 0
        
        values.append(flat) 
        labels[0,index] = label
        index = index + 1
 
    return np.array(values).T, labels


def Split(data, devSize, testSize):
    train = data[...,:-(devSize + testSize)]
    dev = data[...,-(devSize + testSize):-testSize]
    test = data[...,-(testSize):]
    
    return train, dev, test
    
def PrepareData(path, dev_size, test_size):
    X, Y = Load(path)
    trainX, devX, testX = Split(X, dev_size, test_size)
    trainY, devY, testY = Split(Y, dev_size, test_size) 
    
    return trainX, devX, testX, trainY, devY, testY
    
ImagesPath = "C:/Users/Rober/Desktop/Program/Images/Processed"


#trainX, devX, testX, trainY, devY, testY = PrepareData(ImagesPath,50,50)

#print(trainX.shape)
#print(devX.shape)
#print(testX.shape)

#print(trainY.shape)
#print(devY.shape)
#print(testY.shape)






        
        
