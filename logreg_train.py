#!/usr/bin/python3

import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import sys


def hypothesis(x, theta):
    return np.dot(np.transpose(theta),x)


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    a = 1
    return 1 / (1 + np.exp(-x))


def model_optimize(w, b, X, Y):
    m = X.shape[0]

    # Prediction
    final_result = sigmoid(np.dot(w, X.T) + b)
    Y_T = Y.T
    cost = (-1 / m) * (np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
    #

    # Gradient calculation
    dw = (1 / m) * (np.dot(X.T, (final_result - Y.T).T))
    db = (1 / m) * (np.sum(final_result - Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost


def gradientDescent(x, y, theta, m, learning_rate, iterations=1500):
    b = 0
    cost = []
    w = np.zeros((1, 3))
    for iteration in range(iterations):
        theta, cost = model_optimize(w, b, x, y)
        theta[0] = theta["dw"]
        theta[1] = theta["db"]
        # weight update
        w = theta[0] - (learning_rate * (theta[0].T))
        b = theta[1] - (learning_rate * theta[1])
    return theta, cost


def graph(features, output, theta, figure_name):
    x = []
    y = []
    for feature in features:
        x.append(feature[0])
        y.append(feature[1])
    plt.plot(features, output)
    plt.scatter(x, y, c='r', marker='o')
    plt.scatter(x, y, output, c='g', marker='d')
    plt.show()
    plt.savefig(figure_name)


def getFeatures(dataset):
    arithmancy = np.asarray(data['Arithmancy'])
    astronomy = np.asarray(data['Astronomy'])
    herbology = np.asarray(data['Herbology'])
    return (arithmancy, astronomy, herbology)


if __name__ == '__main__':
    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)
    data = dataset.dropna()
    filtered_dataset = data._get_numeric_data()
    theta = np.random.uniform(0.0, 1.0, size=2)
    arithmancy, astronomy, herbology = getFeatures(filtered_dataset)
    houses = np.array(data['Hogwarts House'])
    house_names = ['Ravenclaw', 'Slytherin']#, 'Gryffindor', 'Hufflepuff']
  
    output = [] 
    for x in houses: 
        for y in house_names: 
            if x == y: 
                output.append(house_names.index(y) + 1)

    features = np.asarray([[ar, ast, h] for ar, ast, h in zip(arithmancy, astronomy, herbology)])
    targets = np.asarray([house for house in output])
    alpha = 0.01
    theta = np.random.uniform(0, 1, size=(features.shape[1], 1))
    f = features.shape
    theta, cost = gradientDescent(features[:575], targets, theta, len(targets), alpha)

    plt.plot(cost)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()