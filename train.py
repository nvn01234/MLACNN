import json
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from metrics import F1score
from models import build_model
from settings import *
from utils import make_dict
from sklearn.model_selection import StratifiedKFold
from metrics import evaluate
from itertools import tee
from datetime import datetime


def train(split, x, y, x_index, embeddings, log_dir, model_config={}):
    f1_scores = []
    timestamp = int(datetime.now().timestamp())
    cur_log_dir = "%s/%d" % (log_dir, timestamp)
    os.makedirs(cur_log_dir, exist_ok=True)
    for i, (train_index, test_index) in enumerate(split):
        print("training fold %d" % (i + 1))
        weights_path = "%s/weights_%d.best.h5" % (cur_log_dir, i + 1)

        callbacks = [
            TensorBoard(cur_log_dir),
            F1score(),
            ModelCheckpoint(weights_path, monitor='f1', verbose=1, save_best_only=True, save_weights_only=True,
                            mode='max'),
            EarlyStopping(patience=5, monitor='f1', mode='max')
        ]

        x_train = [d[x_index[train_index]] for d in x]
        y_train = y[train_index]
        x_test = [d[x_index[test_index]] for d in x]
        y_test = y[test_index]
        model = build_model(embeddings, **model_config)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=2, callbacks=callbacks,
                  validation_data=[x_test, y_test])

        print("testing fold %d" % (i + 1))
        model.load_weights(weights_path)
        scores = model.predict(x_test, verbose=False)
        predictions = scores.argmax(-1)
        f1 = evaluate(y_test, predictions, "%s/result_%d.txt" % (cur_log_dir, i + 1))
        print("f1_score: %.2f" % f1)
        f1_scores.append(f1)
    f1_avg = np.average(f1_scores)
    print("model_config: %s, f1_avg = %.2f" % (str(model_config), f1_avg))
    return [timestamp, model_config, f1_avg]


def main():
    print("load data")
    x = [np.load("data/%s.npy" % name) for name in ["words", "pos1", "pos2", "e1", "e2", "e1context", "e2context", "segments"]]
    y = np.load("data/y.npy")
    x_index = np.arange(len(y))
    skf = StratifiedKFold(n_splits=K_FOLD)

    print("load embeddings")
    word_embeddings = np.load("data/embedding/word_embeddings.npy")
    position_embeddings_1 = np.load("data/embedding/position_embeddings_1.npy")
    position_embeddings_2 = np.load("data/embedding/position_embeddings_2.npy")
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2)

    print("training")
    config = K.tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess = K.tf.Session(config=config)
    K.set_session(sess)

    timestamp = int(datetime.now().timestamp())
    log_dir = "log/%d" % timestamp
    split = skf.split(x_index, y)
    split = list(split)

    log_result = [
        train(split, x, y, x_index, embeddings, log_dir, {}),
        train(split, x, y, x_index, embeddings, log_dir, {"lexical_feature": [1, 2, 3, 4]}),
        train(split, x, y, x_index, embeddings, log_dir, {"piecewise_max_pool": True}),
        train(split, x, y, x_index, embeddings, log_dir, {"attention_input": 2}),
        train(split, x, y, x_index, embeddings, log_dir, {"attention_input": 2, "lexical_feature": [1,2,3,4], "piecewise_max_pool": True}),
    ]
    json.dump(log_result, open("%s/log_result.json" % log_dir, "w", encoding="utf8"))

if __name__ == "__main__":
    main()
