#!/usr/bin/python3

import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import sys
from scipy import stats
from scipy.stats import zscore


def hypothesis(x, theta):
    return np.dot(np.transpose(theta),x)


def sigmoid(x):

    # Activation function used to map any real value between 0 and 1
    z = stats.zscore(x)
    return 1 / (1 + np.exp(-z))


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


def gradientDescent(x, y, w, b, m, learning_rate, iterations=15000):
    costs = []
    for iteration in range(iterations):
        grads, cost = model_optimize(w, b, x, y)
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * dw.T)
        b = b - (learning_rate * db)
        if iteration % 100 == 0:
            costs.append(cost)
    coeff = {"w": w, "b": b}
    return coeff, costs

"""
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
"""


def getFeatures(dataset):
    arithmancy = np.asarray(data['Arithmancy'])
    astronomy = np.asarray(data['Astronomy'])
    herbology = np.asarray(data['Herbology'])
    return arithmancy, astronomy, herbology


def variables_initialization(features):
    alpha = 0.0001
    b = 0
    w = np.random.uniform(0, 1, features.shape[1])
    coeffs = []
    return alpha, b, w, coeffs


if __name__ == '__main__':

    ''' Parser '''

    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)
    data = dataset.dropna()
    filtered_dataset = data._get_numeric_data()
    arithmancy, astronomy, herbology = getFeatures(filtered_dataset)
    houses = np.array(data['Hogwarts House'])
    house_names = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

    ''' Parse Houses '''

    output = [] 
    for x in houses: 
        for y in house_names: 
            if x == y: 
                output.append(house_names.index(y) + 2)

    ''' Parse Features '''

    features = np.asarray([[ar, ast, h] for ar, ast, h in zip(arithmancy, astronomy, herbology)])
    features_na = np.nan_to_num(features)
    targets = np.asarray([house for house in output])

    ''' Launch Gradient '''

    alpha, b, w, coeffs = variables_initialization(features)
    coeffs, costs = gradientDescent(features_na[:1251], targets, w, b, 1251, alpha)

    ''' Final prediction '''

    w = coeffs["w"]
    b = coeffs["b"]
    print('Optimized weights', w)
    print('Optimized intercept', b)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()
