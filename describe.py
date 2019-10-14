#!/usr/bin/python3

"""Functional program allowing the user to visualize the numerical values of a CSV passed as an argument"""

import pandas as pd
import numpy as np
import math
import sys

if __name__ == '__main__':
    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)

    filtered_dataset = dataset.dropna(axis='columns', how='all')
    filtered_dataset = filtered_dataset._get_numeric_data()
    features = []
    for col in dataset.columns:
        try:
            float(dataset[col][0])
            features.append(col)
        except ValueError:
            continue
    info = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Unique", "Top", "Freq", "NaN"]
    df = pd.DataFrame(np.zeros(shape=(len(info), len(features))), index = info, columns = features)
    for col_name in df.columns:
        n = 0
        total = 0
        _max = - np.inf
        _min = np.inf
        ordered_list = []
        count = 0
        for val in dataset[col_name]:
            if np.isnan(val):
                count += 1
                continue
            else:
                ordered_list.append(val)
                n += 1
                total = total + val
                if _max < val:
                    _max = val
                if _min > val:
                    _min = val
            if n == 0:
                df.drop(labels=col_name, axis=1)
            else:
                ordered_list.sort()
                df[col_name]["Count"] = n
                df[col_name]["Mean"] = total / n
                df[col_name]["Max"] = _max
                df[col_name]["Min"] = _min
                df[col_name]["Std"] = np.nanstd(dataset[col_name])
                df[col_name]["25%"] = ordered_list[math.ceil(n / 4) - 1]
                df[col_name]["50%"] = ordered_list[math.ceil(n / 2) - 1]
                df[col_name]["75%"] = ordered_list[math.ceil(3 * n / 4) - 1]
                df[col_name]['Unique'] = len(ordered_list)
                temp = [[ordered_list.count(x), x] for x in set(ordered_list)]
                temp.sort()
                tempmax = temp[len(temp) - 1]
                df[col_name]['Top'] = tempmax[1]
                df[col_name]['Freq'] = tempmax[0]
                df[col_name]['NaN'] = count

    print(df)
