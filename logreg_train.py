import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import sys

def hypothesis(x, theta):
	return np.dot(np.transpose(theta),x)

def gradientDescent(x, y, theta, m, alpha, iterations=1500):
	for iteration in range(iterations):
		for j in range(len(theta)):
			gradient = 0
			for i in range(m):
				gradient += (hypothesis(x[i], theta) - y[i]) * x[i][j]
		gradient *= 1/m
		theta[j] = theta[j] - (alpha * gradient)
	return theta

def graph(features, output, theta, figure_name):
    x = []
    y = []
    for feature in features:
        x.append(feature[0])
        y.append(feature[1])
    plt.plot(features, output)
    plt.scatter(x, y, c='r', marker='o')
    plt.scatter(x, y, output, c='g', marker='d')
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
    house_names = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
  
    output = [] 
    for x in houses: 
        for y in house_names: 
            if x == y: 
                output.append(house_names.index(y)) 

    features = np.asarray([[ar, ast, h] for ar, ast, h in zip(arithmancy, astronomy, herbology)])
    outputs = np.asarray([house for house in output])
    alpha = 0.01
    theta = np.random.uniform(0.0, 1.0, size=3)

    theta = gradientDescent(features, outputs, theta, len(outputs), alpha)
    graph(features, outputs, theta, 'graphPostFit')
