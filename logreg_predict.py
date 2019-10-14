import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import sys


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0.2)
    """
    data = dataset.iloc[:,6:]
    weights = pd.read_csv("weights.csv", delimiter=",")
    w = np.asarray(weights)

    X = np.nan_to_num(data)
    means = column_means(X)
    stdevs = column_stdevs(X, means)
    standardize_dataset(X, means, stdevs)
    Y = []
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    for i in range(len(houses)):
        y_pred = sigmoid(np.dot(w[i], X.T))
        Y.append(y_pred)
    Y_T = np.array(Y).T

    with open("houses.csv","w+") as f:
        f.write('Index,Hogwarts House\n')
        for i, row in enumerate(Y_T):
            pred = houses[row.argmax()]
            f.write('{0},{1}\n'.format(i, pred))
