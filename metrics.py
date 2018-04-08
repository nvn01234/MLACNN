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

        scores = self.model.predict(self.validation_data[:3], verbose=False)
        predictions = scores.argmax(-1)
        y_true = self.validation_data[3]
        f1 = evaluate(y_true, predictions)
        print(' - f1: {:04.2f}'.format(f1))
        logs['f1'] = f1

def gen_key(path, data, idx2relations):
    with open(path, "w", encoding="utf8") as f:
        for idx, y in enumerate(data):
            f.write("%s\t%s\n" % (idx, idx2relations[y]))


def evaluate(y_true, y_pred, result_path=None):
    y_pred = np.reshape(y_pred, (-1,))
    y_true = np.reshape(y_true, (-1,))
    idx2relations = {}
    with open("origin_data/relations.txt", "r", encoding="utf8") as f:
        for line in f:
            idx, r = line.strip().split()
            idx2relations[int(idx)] = r

    log_dir = "log"
    test_keys_path = os.path.join(log_dir, "test_keys.txt")
    predict_keys_path = os.path.join(log_dir, "predict_keys.txt")

    os.makedirs(log_dir, exist_ok=True)
    gen_key(test_keys_path, y_true, idx2relations)
    gen_key(predict_keys_path, y_pred, idx2relations)

    if result_path is None:
        result_path = os.path.join(log_dir, "tmp_result.txt")
        keep_result = False
    else:
        keep_result = True

    os.system("perl scorer.pl %s %s > %s" % (predict_keys_path, test_keys_path, result_path))
    with open(result_path, "r") as f:
        f1 = float(f.read().strip()[-10:-5])

    os.remove(test_keys_path)
    os.remove(predict_keys_path)
    if not keep_result:
        os.remove(result_path)

    return f1

