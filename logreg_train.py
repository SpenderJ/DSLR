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

def next_batch(X, y, batchSize):
	# loop over our dataset `X` in mini-batches of size `batchSize`
	for i in np.arange(0, X.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (X[i:i + batchSize], y[i:i + batchSize])

def stocashtic_gradient_descent(X, Y, w, b, m, learning_rate=0.01, iterations=10):
    costs = []
    errors = []
    for iteration in range(iterations):
        cost = 0.0
        for (batchX, batchY) in next_batch(X, Y, 20):
            grads, cost, error = model_optimize(w, b, batchX, batchY)
            dw = grads["dw"]
            db = grads["db"]
            # weight update
            w = w - (learning_rate * dw.T)
            b = b - (learning_rate * db)
        #if iteration % 15 == 0:
        costs.append(cost)
        errors.append(error)
    coeff = {"w": w, "b": b}
    return coeff, costs, errors

def gradientDescent(X, Y, w, b, m, learning_rate, iterations=1500):
    costs = []
    errors = []
    for iteration in range(iterations):
        grads, cost, error = model_optimize(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * dw.T)
        b = b - (learning_rate * db)
        if iteration % 15 == 0:
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


''' Retrieves features for each four houses '''
def getHouses(dataset, house_names):
    houses = []
    Y_= np.asarray(dataset['Hogwarts House'])
    for x in Y_: 
        for y in house_names: 
            if x == y: 
                houses.append(house_names.index(y))
    return houses


def getFeatures(dataset):
    feats = np.asarray(dataset.iloc[:,6:])
    features = np.nan_to_num(feats)
    means = column_means(features)
    stdevs = column_stdevs(features, means)
    standardize_dataset(features, means, stdevs)
    return features


if __name__ == '__main__':

    ''' Parser '''

    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)
    """
    X = data2.iloc[:,1:]
    y = data2.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    ''' Standardize features '''
    features = getFeatures(dataset)
    house_names = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    houses = getHouses(dataset, house_names)
    cost_hist = []
    weights = []
    for i in range(len(house_names)):
        t = []
        for j in houses:
            if (j == i):
                t.append(1)            
            else:
                t.append(0)
        targets = np.asarray(t)
        alpha, b, w, coeffs = variables_initialization(features)

        ''' Start logistic regression '''
        coeffs, costs, errors = stocashtic_gradient_descent(features, targets, w, b, len(features), alpha)
        weights.append(coeffs)

    ''' Final prediction '''
    with open("weights.csv","w+") as f:
        f.write('0,1,2,3,4,5,6,7,8,9,10,11,12\n')
        for i in range(len(house_names)):
            np.savetxt(f, weights[i]['w'], delimiter=",")


    print('Optimized weights', weights)
    print('Final cost:', costs)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.plot(errors)
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()

