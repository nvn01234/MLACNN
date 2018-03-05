import numpy as np
from settings import *
import math
from utils import make_dict


def main():
    y_true = np.load("data/test/labels.npy")
    y_pred = np.load("log/y_pred_3.npy")

    # confusion matrix
    cm19 = np.zeros([NB_RELATIONS, NB_RELATIONS], dtype="int32")
    cm10 = np.zeros([NB_UND_RELATIONS, NB_UND_RELATIONS], dtype="int32")
    for i, j in zip(y_true, y_pred):
        cm19[i, j] += 1
        cm10[math.floor(i/2), math.floor(j/2)] += 1

    metrics_19 = metrics(cm19, [-1, 7])
    metrics_10 = metrics(cm10, [-1])

    print("")


def metrics(cm, exclude):
    sum_col = cm.sum(1)
    sum_row = cm.sum(0)
    total = sum_row.sum()
    precisions = cm.diagonal() / sum_row
    recalls = cm.diagonal() / sum_col
    f1s = f1_score(precisions, recalls)
    micro_precision = np.delete(cm.diagonal(), exclude).sum() / np.delete(sum_row, exclude).sum()
    micro_recall = np.delete(cm.diagonal(), exclude).sum() / np.delete(sum_col, exclude).sum()
    micro_f1 = f1_score(micro_precision, micro_recall)
    macro_precision = np.average(np.delete(precisions, exclude))
    macro_recall = np.average(np.delete(recalls, exclude))
    macro_f1 = f1_score(macro_precision, macro_recall)
    return make_dict(sum_col, sum_row, total, precisions, recalls, f1s, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1)


def f1_score(precision, recall):
    return 2 / (1 / precision + 1 / recall)


if __name__ == "__main__":
    main()