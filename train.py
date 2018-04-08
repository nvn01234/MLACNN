import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow import ConfigProto, Session
from keras import backend as K
from models import build_model
from settings import *
from test import evaluate
from utils import make_dict
from sklearn.model_selection import StratifiedKFold


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

    if not os.path.exists("model"):
        os.makedirs("model")

    split = skf.split(x_index, y)
    f1_scores = []
    for i, (train_index, test_index) in enumerate(split):
        print("training fold %d" % (i+1))
        filepath = "model/weights_%d.best.hdf5" % (i+1)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        x_train = [d[x_index[train_index]] for d in x]
        y_train = y[train_index]
        x_test = [d[x_index[test_index]] for d in x]
        y_test = y[test_index]
        model = build_model(embeddings)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True, callbacks=[checkpoint], validation_data=[x_test, y_test])

        print("testing fold %d" % (i+1))
        model.load_weights(filepath)
        scores = model.predict(x_test, verbose=False)
        predictions = scores.argmax(-1)
        f1_score = evaluate(predictions, y_test, i+1)
        f1_scores.append(f1_score)
    f1_avg = sum(f1_scores) / len(f1_scores)
    print("f1_avg = %.2f" % f1_avg)

if __name__ == "__main__":
    main()
