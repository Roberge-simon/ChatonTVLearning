import plotter
import DataLoader

import numpy as np
import itertools


ALPHA= 0.001
LAMBDA = 0.1

def Evaluate(input, w, b):
    linear = (np.dot(w.T, input) + b)    
    sigmoid = 1 / (1 + np.exp(-1 * linear))
    return sigmoid

#-ylog( hθ(x) ) - (1-y)log( 1- hθ(x) ) 
def Cost(result, labels, w, b):
    
    cost = np.average( (-labels * np.log(result)) - ((1-labels)*np.log(1-result)))
    
    m =  labels.shape[1]
    reg = LAMBDA/(2*m) * np.linalg.norm(w)
    cost = cost + reg
    
    return cost
    
def EvaluateCost(input, labels, w, b):
    return Cost(Evaluate(input, w,b), labels, w, b)

def Predict(input, w, b):
    result = Evaluate(input, w, b)
    prediction = np.greater(result, 0.5)
    return prediction
    
def CheckPrediction(input, labels, w , b):
    prediction = Predict(input, w, b)
    accuracy = np.average(np.equal(prediction, labels)) *100
    return accuracy
    
def Assess(result, labels):
    return False
    
def Improve(w, b, input, result, labels):
    dw, db = ComputeGrads(input, result, labels, w, b)
    
    #_dw, _db = ApproxGrads(input, labels, w, b)
    #print("dw :" + str(dw) + "\n approx : ", str(_dw))
    #print("db :" + str(db) + "\n approx : ", str(_db))
    
    #assert np.linalg.norm(dw - _dw) < 1e-4
    #assert np.linalg.norm(db - _db) < 1e-4

    return w - ALPHA * dw, b - ALPHA * db
    
def ApproxGrads(input, labels, w, b):
    epsilon = 1e-4
    m = input.shape[1]
    
    dw = np.zeros((input.shape[0],1))
    for i in range(input.shape[0]):
        _wPlus = w * 1
        _wPlus[i] += epsilon
        _wMinus = w * 1
        _wMinus[i] -= epsilon 
        dw[i] = (EvaluateCost(input, labels, _wPlus, b) - EvaluateCost(input, labels, _wMinus, b))/ (2*epsilon)
        
    db = (EvaluateCost(input, labels, w, b + epsilon) - EvaluateCost(input, labels, w, b - epsilon))/ (2*epsilon)
    
    return dw, db
    
    
def ComputeGrads(input, result, labels, w , b):
    m = input.shape[1]
    dz = result - labels
    
    dw = np.dot(input, dz.T) / m
    db = np.sum(dz, axis = 1, keepdims = True) / m
    
    reg = (LAMBDA / m) * w
    dw = reg + dw
 
    return dw, db
   
    
def TrainOne(input, labels, w, b):
    result = Evaluate(input, w, b)
    cost = Cost(result, labels,w, b)
    w, b = Improve(w, b,input, result, labels)
    return w, b, cost
    
def TrainMany(input, labels, w, b):
    _w = w
    _b = b
    while True:
        _w, _b , cost = TrainOne(input, labels, _w, _b) 
        yield _w, _b, cost
        
def TrainModel(trainInput, trainLabels, epochs):
    nx = trainInput.shape[0]
    w = np.random.random((nx,1)) * 0.001
    b = np.zeros((1,1))
    
    trainer = TrainMany(trainInput, trainLabels, w, b)
    plot = plotter.Plotter()
    plot.Start(epochs)
    for _w,_b, cost in itertools.islice(trainer, epochs):
        print ("Cost: " + str(cost))
        plot.Update(cost)
    return _w, _b

def Test(input, labels, w, b):
    validity = CheckPrediction(input, labels, w, b)
    print("Validity: " +str(validity))
    
    
def DummyData(nx, m):
    x = np.random.randn(nx, m)
    y = np.ones((1, m))
    return x, x, x, y, y, y
    
trainX, devX, testX, trainY, devY, testY = DataLoader.PrepareData(DataLoader.ImagesPath, 50, 50)
#trainX, devX, testX, trainY, devY, testY = DummyData(3, 10)
#print(trainX)

w, b = TrainModel(trainX, trainY, 3000)
Test(devX, devY, w, b)
#Test(testX, testY, w, b)
