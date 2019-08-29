#!/usr/bin/python3

"""Program checking which 2 features look the same in the dataset"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


input_file = "dataset/dataset_train.csv"

if __name__ == '__main__':
    try:
        dataset = pd.read_csv(input_file, delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)

    Ravenclaw = dataset[(dataset['Hogwarts House'] == 'Ravenclaw')]
    Slytherin = dataset[(dataset['Hogwarts House'] == 'Slytherin')]
    Gryffindor = dataset[(dataset['Hogwarts House'] == 'Gryffindor')]
    Hufflepuff = dataset[(dataset['Hogwarts House'] == 'Hufflepuff')]

    Ravenclaw = Ravenclaw.dropna(axis='columns', how='all')
    Ravenclaw = Ravenclaw._get_numeric_data()
    Slytherin = Slytherin.dropna(axis='columns', how='all')
    Slytherin = Slytherin._get_numeric_data()
    Gryffindor = Gryffindor.dropna(axis='columns', how='all')
    Gryffindor = Gryffindor._get_numeric_data()
    Hufflepuff = Hufflepuff.dropna(axis='columns', how='all')
    Hufflepuff = Hufflepuff._get_numeric_data()

    ax = Ravenclaw.plot.scatter(x='Potions', y='Care of Magical Creatures', color='red', label='Ravenclaw')
    ax = Slytherin.plot.scatter(x='Potions', y='Care of Magical Creatures', color='black', label='Slytherin', ax=ax)
    ax = Gryffindor.plot.scatter(x='Potions', y='Care of Magical Creatures', color='saddlebrown', label='Gryffindor', ax=ax)
    Hufflepuff.plot.scatter(x='Potions', y='Care of Magical Creatures', color='green', label='Hufflepuff', ax=ax)
    plt.show()
