import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import sys

if __name__ == '__main__':

    ''' Parser '''

    try:
        dataset = pd.read_csv(sys.argv[1], delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)