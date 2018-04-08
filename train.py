import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow import ConfigProto, Session
from keras import backend as K

from metrics import F1score
from models import build_model
from settings import *
from utils import make_dict
from sklearn.model_selection import StratifiedKFold
from metrics import evaluate


def main():
    print("load data")
    x = [np.load("data/%s.npy" % name) for name in ["words", "pos1", "pos2"]]
    y = np.load("data/y.npy")
    x_index = np.arange(len(y))
    skf = StratifiedKFold(n_splits=K_FOLD)

    print("load embeddings")
    word_embeddings = np.load("data/embedding/word_embeddings.npy")
    position_embeddings_1 = np.load("data/embedding/position_embeddings_1.npy")
    position_embeddings_2 = np.load("data/embedding/position_embeddings_2.npy")
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2)

    print("training")
    config = ConfigProto()
    config.log_device_placement = False
    # config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)

    os.makedirs("model", exist_ok=True)
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    split = skf.split(x_index, y)
    f1_scores = []
    for i, (train_index, test_index) in enumerate(split):
        print("training fold %d" % (i+1))
        weights_path = "model/weights_%d.best.h5" % (i+1)

        callbacks = [
            TensorBoard(log_dir),
            F1score(),
            ModelCheckpoint(weights_path, monitor='f1', verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
            EarlyStopping(monitor='f1', patience=5, mode='max')
        ]

        x_train = [d[x_index[train_index]] for d in x]
        y_train = y[train_index]
        x_test = [d[x_index[test_index]] for d in x]
        y_test = y[test_index]
        model = build_model(embeddings)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True, callbacks=callbacks, validation_data=[x_test, y_test])

        print("testing fold %d" % (i+1))
        model.load_weights(weights_path)
        scores = model.predict(x_test, verbose=False)
        predictions = scores.argmax(-1)
        f1 = evaluate(y_test, predictions, "log/result_%d.txt" % (i + 1))
        print("f1_score: %.2f" % f1)
        f1_scores.append(f1)
    f1_avg = sum(f1_scores) / len(f1_scores)
    print("f1_avg = %.2f" % f1_avg)

if __name__ == "__main__":
    main()
