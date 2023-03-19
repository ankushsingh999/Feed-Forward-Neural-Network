# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:49:14 2023

@author: Ankush Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 80
NUM_OUTPUT = 10


def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weights[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def relu(x) :
    relu = np.maximum(0,x)
    return relu  

def relu_dash(input):
    relu_d = input.copy()
    relu_d[relu_d <= 0] = 0
    relu_d[relu_d > 0] = 1
    return relu_d

def accuracy(yh,Y):
    g_truth = np.argmax(Y, axis=0)      
    pred = np.argmax(yh, axis=0)    
    accuracy = np.sum(g_truth == pred)/Y.shape[1]            
    return accuracy



def softmax(z):
    znormed = z - np.max(z, axis=0, keepdims=True)
    denom = np.sum(np.exp(znormed), axis=0, keepdims=True)
    result = np.exp(znormed) / denom
    return result

def fCE(X, Y, weights):
    
    Ws, bs = unpack(weights)
    # ...
    #for the first layer
    h = X
    #pre activation
    prZ = []
    #post activation
    pZ = []
    for i in range(NUM_HIDDEN_LAYERS):
        b = bs[i].reshape(-1, 1)
        Z = np.dot(Ws[i], h) + b
        prZ.append(Z)
        h = relu(Z)
        pZ.append(h)
        
    Lg = np.dot(Ws[-1], pZ[-1]) + bs[-1].reshape(-1, 1)
    prZ.append(Lg)
    yhat = softmax(Lg)
    #cost
    cost = -np.sum(np.log(yhat) * Y)/Y.shape[1]
    #accuracy of the model
    acc = accuracy(yhat,Y)
    #print("accuracy:",acc)
    return cost,prZ, pZ,yhat,acc 
   
def gradCE (X, Y, weights):

    loss, h_z, h_h, yhat,_ = fCE(X, Y, weights)
    #store gradCE of the weights
    SgW = []  
    #store gradCE of the biases
    Sgb = []  
    Ws, bs = unpack(weights)
    grad = yhat - Y

    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        #gradCE for biases
        #for last layer
        if i != NUM_HIDDEN_LAYERS:  
            dz = relu_dash(h_z[i])
            grad = dz * grad
        gb = np.sum(grad, axis=1) / Y.shape[1]
        Sgb.append(gb)
        #for the first layer in the network
        if i == 0:
            SgW.append(np.dot(grad, X.T) / Y.shape[1])
        else:
            SgW.append(np.dot(grad, h_h[i - 1].T) / Y.shape[1])   
        grad = np.dot(Ws[i].T, grad) 

    SgW.reverse()  
    Sgb.reverse()  
    
    return np.hstack([GW.flatten() for GW in SgW] + [GB.flatten() for GB in Sgb])
   


def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN ** 0.5)
    plt.imshow(np.vstack([
    np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()
    
def train (trainX, trainY, weights, testX, testY):
    change_order_index = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_index]
    trainY = trainY[:, change_order_index]

    index_values = np.random.permutation(trainX.shape[1])
    train_X = trainX[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_X = trainX[:, index_values[int(trainX.shape[1] * 0.8):]]
    train_Y = trainY[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_Y = trainY[:, index_values[int(trainX.shape[1] * 0.8):]]
    
    #e = [50,100,150]
    #lr = [0.001,0.005]
    #mini = [80,90,128]
    inicost = 10000000000000000
    e = [100]
    lr = [0.005]
    mini = [128]
    #print("Epochs: ", e, "Learning Rate: ", lr, "Mini Batch Size: ",mini)
    for ep in e:
        for lrr in lr:
            for minin in mini:
                print("Epochs: ", ep, "Learning Rate: ", lrr, "Mini Batch Size: ",minin)
                for ne in range(ep):
                    batches = int((len(train_X.T) / minin))
                    start = 0
                    end = minin
                    for j in range(batches):
                        xminib = train_X[:,start:end]
                        yminib = train_Y[:,start:end]
                        #print("shape of xminib:", np.shape(xminib), np.shape(yminib))
                        grad = gradCE(xminib,yminib,weights)
                        weights = weights -lrr*grad
                        start = end
                        end = end+minin
                cost,_,_,_,acc = fCE(valid_X,valid_Y,weights)
                if cost < inicost:
                    bestw = weights
                    weights = bestw
                    
    cost,_,_,_,acc = fCE(testX,testY,bestw)
    print("cost",cost)
    print("Accuracy of the model",acc*100)
    return weights

            

def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs

if __name__ == "__main__":
    # Load training data.
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).
    if "trainX" not in globals():
        trainX = np.load("fashion_mnist_train_images.npy").T/255.
        trainy = np.load("fashion_mnist_train_labels.npy")
        testX = np.load("fashion_mnist_test_images.npy").T /255.
        testy = np.load("fashion_mnist_test_labels.npy")
    
    Ws, bs = initWeightsAndBiases()
    
    trainY = np.zeros((trainy.size, trainy.max() + 1))
    testY = np.zeros((testy.size, testy.max() + 1))
    trainY[np.arange(trainy.size), trainy] = 1
    testY[np.arange(testy.size), testy] = 1
    trainY = trainY.T
    testY = testY.T
    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).    
    print(scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_)[0], \
                                    lambda weights_: gradCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_), \
                                    weights))
    print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_)[0], 1e-6))

    weights = train(trainX, trainY, weights, testX, testY)
    
    show_W0(weights)