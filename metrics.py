import json

from keras.callbacks import Callback
import os
import time
import numpy as np


class F1score(Callback):

    def __init__(self):
        super(F1score, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        scores = self.model.predict(self.validation_data[:-3], verbose=False)
        predictions = scores.argmax(-1)
        y_true = self.validation_data[-3]
        f1 = evaluate(y_true, predictions)
        print(' - f1: {:04.2f}'.format(f1))
        logs['f1'] = f1

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def evaluate(y_true, y_pred, result_path=None):
    y_pred = np.reshape(y_pred, (-1,))
    y_true = np.reshape(y_true, (-1,))

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    with open("origin_data/relations.txt", "r", encoding="utf8") as f:
        nlabels = len(f.readlines())

    matrix = np.zeros([nlabels, nlabels], dtype='int32')
    for i, j in zip(y_true, y_pred):
        matrix[i, j] += 1
    sum_col = matrix.sum(1)
    sum_row = matrix.sum(0)
    sum_all = np.sum(matrix)
    precisions = matrix.diagonal() / sum_row
    recalls = matrix.diagonal() / sum_col
    f1s = f1_score(precisions, recalls)

    micro_precision = np.sum(matrix.diagonal()[:-1]) / np.sum(sum_row[:-1])
    micro_recall = np.sum(matrix.diagonal()[:-1]) / np.sum(sum_col[:-1])
    micro_f1 = f1_score(micro_precision, micro_recall)
    macro_precision = np.average(precisions[:-1])
    macro_recall = np.average(recalls[:-1])
    macro_f1 = np.average(f1s[:-1])

    if result_path is not None:
        log = {
            "y_pred": y_pred.tolist(),
            "y_true": y_true.tolist(),
            "matrix": matrix.tolist(),
            "sum_col": sum_col.tolist(),
            "sum_row": sum_row.tolist(),
            "sum_all": sum_all,
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
            "f1s": f1s.tolist(),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }
        with open(result_path, "w", encoding="utf8") as f:
            json.dump(log, f, ensure_ascii=False)

    return macro_f1

