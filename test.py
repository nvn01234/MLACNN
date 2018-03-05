import numpy as np
from settings import *
import math


def main():
    y_true = np.load("data/test/labels.npy")
    y_pred = np.load("log/y_pred.npy")

    # confusion matrix
    cm19 = np.zeros([NB_RELATIONS, NB_RELATIONS])
    cm10 = np.zeros([NB_UND_RELATIONS, NB_UND_RELATIONS])
    for i, j in zip(y_true, y_pred):
        cm19[i, j] += 1
        cm10[math.ceil(i/2), math.ceil(j/2)] += 1
    print("")



if __name__ == "__main__":
    main()