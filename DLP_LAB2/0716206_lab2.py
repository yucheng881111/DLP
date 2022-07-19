# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:20:30 2022

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(4)

def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[1]):
        if y[0, i] == 0:
            plt.plot(x[0, i], x[1, i], 'ro')
        else:
            plt.plot(x[0, i], x[1, i], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[1]):
        if abs(pred_y[0, i] - 1) > abs(pred_y[0, i]):
            plt.plot(x[0, i], x[1, i], 'ro')
        else:
            plt.plot(x[0, i], x[1, i], 'bo')
    
    plt.show()
    
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def ReLU(x):
    x = np.maximum(0, x)
    return x

def derivative_ReLU(x):
    x[x <= 0] = 0
    return x
    
def forward(a0, w1, w2, w3):
    z1 = np.dot(w1, a0)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    z3 = np.dot(w3, a2)
    a3 = sigmoid(z3)
    
    return a3, a0, a1, a2, a3
    
def cost(pred_y, y):
    # cross entropy
    return float(-(1/y.shape[1])*(y @ np.log(pred_y + 0.0001).T+(1-y) @ np.log(1-pred_y + 0.0001).T))

def backpropagation(pred_y, y, learning_rate, n, a0, w1, a1, w2, a2, w3, a3):
    dJ_da3 = -(y / (pred_y + 0.0001) - (1 - y) / (1 - pred_y + 0.0001))
    dJ_dz3 = derivative_sigmoid(a3) * dJ_da3
    dJ_dw3 = np.dot(dJ_dz3, a2.T)
    
    dJ_da2 = np.dot(w3.T, dJ_dz3)
    dJ_dz2 = derivative_sigmoid(a2) * dJ_da2
    dJ_dw2 = np.dot(dJ_dz2, a1.T)
    
    dJ_da1 = np.dot(w2.T, dJ_dz2)
    dJ_dz1 = derivative_sigmoid(a1) * dJ_da1
    dJ_dw1 = np.dot(dJ_dz1, a0.T)
    
    w1 = w1 - learning_rate * (dJ_dw1 / n)
    w2 = w2 - learning_rate * (dJ_dw2 / n)
    w3 = w3 - learning_rate * (dJ_dw3 / n)
    
    return w1, w2, w3


if __name__=='__main__':
    x, y = generate_linear()
    #x, y = generate_XOR_easy()
    x = x.T
    y = y.T
    
    n = 10 # weight size
    w1 = np.random.randn(n, 2)
    w2 = np.random.randn(n, n)
    w3 = np.random.randn(1, n)
    
    # train
    print('training...')
    for i in range(5000):
        pred_y, a0, a1, a2, a3 = forward(x, w1, w2, w3)
        loss = cost(pred_y, y)
        w1, w2, w3 = backpropagation(pred_y, y, 0.1, n, a0, w1, a1, w2, a2, w3, a3)
        if i % 10 == 0:
            print('epoch ' + str(i) + ' loss: ' + str(cost(pred_y, y)))
            #accuracy = (1 - np.sum(np.abs(y - np.round(pred_y))) / y.shape[1]) * 100
            #print('accuracy: ' + str(accuracy) + '%')
    
    # test
    print('\ntesting...')
    x_test, y_test = generate_linear()
    #x_test, y_test = generate_XOR_easy()
    x_test = x_test.T
    y_test = y_test.T
    pred_y, a0, a1, a2, a3 = forward(x_test, w1, w2, w3)
    accuracy = (1 - np.sum(np.abs(y_test - np.round(pred_y))) / y_test.shape[1]) * 100
    print(pred_y)
    print('accuracy: ' + str(accuracy) + '%')
    show_result(x_test, y_test, pred_y)
    






