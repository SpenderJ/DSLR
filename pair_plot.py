#!/usr/bin/python3

"""Program creating  pair plot of all the numeric values contained in the dataset inputted"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

input_file = "dataset/dataset_train.csv"

if __name__ == '__main__':
    try:
        dataset = pd.read_csv(input_file, delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)

    filtered_dataset = dataset.dropna(axis='columns', how='all')
    filtered_dataset = filtered_dataset._get_numeric_data()

    sns.pairplot(filtered_dataset)
    plt.show()