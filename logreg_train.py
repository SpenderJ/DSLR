#!/usr/bin/python3

import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import sys
from scipy import stats
from scipy.stats import zscore

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def model_optimize(w, b, X, Y):
    m = X.shape[0]
    # Prediction
    final_result = sigmoid(np.dot(w, X.T) + b)
    Y_T = Y.T
    cost = (-1 / m) * (np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
    error = np.mean(final_result - Y.T)
    dw = (1 / m) * (np.dot(X.T, (final_result - Y.T).T))
    db = (1 / m) * (np.sum(final_result - Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost, error


def gradientDescent(X, Y, w, b, m, learning_rate, iterations=15000):
    costs = []
    errors = []
    for iteration in range(iterations):
        grads, cost, error = model_optimize(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * dw.T)
        b = b - (learning_rate * db)
        if iteration % 150 == 0:
            costs.append(cost)
            errors.append(error)
    coeff = {"w": w, "b": b}
    return coeff, costs, errors

def variables_initialization(features):
    alpha = 0.01
    b = 0
    w = np.zeros((1, features.shape[1]))
    coeffs = []
    return alpha, b, w, coeffs


# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means
 
# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [np.sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]

def normalize_dataset(dataset):
    for row in dataset:
        maxX = max(row)
        minX = min(row)
        for i in range(len(row)):
            row[i] = (row[i] - maxX) / (maxX - minX)

if __name__ == '__main__':

    ''' Parser '''

    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)

    #data = dataset.dropna()
    feats = dataset.iloc[:,6:]
    features = np.nan_to_num(feats)
    #normalize_dataset(features)
    print(features)
    means = column_means(features)
    print(means)
    stdevs = column_stdevs(features, means)
    print(stdevs)
    standardize_dataset(features, means, stdevs)
    print(features)
    houses = np.array(dataset['Hogwarts House'])
    house_names = ['Ravenclaw']#, 'Slytherin', 'Gryffindor', 'Hufflepuff']

    ''' Parse Houses '''

    output = [] 
    for x in houses: 
        for y in house_names: 
            if x == y: 
                output.append(house_names.index(y))

    ''' Parse Features '''

    targets = np.asarray([house for house in output])
    ''' Launch Gradient '''

    alpha, b, w, coeffs = variables_initialization(features)
    coeffs, costs, errors = gradientDescent(features[:443], targets, w, b, len(targets), alpha)

    ''' Final prediction '''

    w = coeffs["w"]
    b = coeffs["b"]
    print('Optimized weights', w)
    print('Optimized intercept', b)

    plt.plot(costs)
    plt.ylabel('cost')
    #plt.plot(errors)
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()
