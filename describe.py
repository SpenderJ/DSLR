#!/usr/bin/python3

"""Functional program allowing the user to visualize the numerical values of a CSV passed as an argument"""

import pandas as pd
import sys

# Program

if __name__ == '__main__':
    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)

    print("Success let's go to the next step")
    filtered_dataset = dataset.dropna(axis='columns', how='all')
    filtered_dataset = filtered_dataset._get_numeric_data()
    print(filtered_dataset)