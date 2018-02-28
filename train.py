import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow import ConfigProto
from settings import *
from models import build_model
from sklearn.metrics import f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = ConfigProto()
config.log_device_placement = False
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def main():
    print("loading data")
    x_words_train = np.load("data/x_words_train.npy")
    x_pos1_train = np.load("data/x_pos1_train.npy")
    x_pos2_train = np.load("data/x_pos2_train.npy")
    x_e1_train = np.load("data/x_e1_train.npy")
    x_e1_train = np.reshape(x_e1_train, (-1, 1))
    x_e2_train = np.load("data/x_e2_train.npy")
    x_e2_train = np.reshape(x_e2_train, (-1, 1))
    y_train = np.load("data/y_train.npy")

    print("load testing data")
    x_words_test = np.load("data/x_words_test.npy")
    x_pos1_test = np.load("data/x_pos1_test.npy")
    x_pos2_test = np.load("data/x_pos2_test.npy")
    x_e1_test = np.load("data/x_e1_test.npy")
    x_e1_test = np.reshape(x_e1_test, (-1, 1))
    x_e2_test = np.load("data/x_e2_test.npy")
    x_e2_test = np.reshape(x_e2_test, (-1, 1))
    y_test = np.load("data/y_test.npy")

    print("training")
    model = build_model()
    model.fit([x_words_train, x_pos1_train, x_pos2_train, x_e1_train, x_e2_train], y_train, epochs=NB_EPOCHS, verbose=True, batch_size=BATCH_SIZE)
    scores = model.predict([x_words_test, x_pos1_test, x_pos2_test, x_e1_test, x_e2_test], verbose=False)
    y_pred = scores.argmin(axis=-1)
    f1 = f1_score(y_test, y_pred, average="macro")
    print("f1-macro: %.4f%%" % (f1*100))


if __name__ == "__main__":
    main()