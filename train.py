import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow import ConfigProto, Session
from keras import backend as K
from models import build_model
from settings import *
from test import gen_answer_key
from utils import make_dict


def main():
    print("load train data")
    x_train = [np.load("data/train/%s.npy" % name) for name in ["words", "pos1", "pos2", "tags", "chars", "e1", "e2", "e1context", "e2context"]]
    y_train = np.load("data/train/y.npy")

    print("load test data")
    x_test = [np.load("data/test/%s.npy" % name) for name in ["words", "pos1", "pos2", "tags", "chars", "e1", "e2", "e1context", "e2context"]]
    y_test = np.load("data/test/y.npy")

    print("load embeddings")
    word_embeddings = np.load("data/embedding/word_embeddings.npy")
    position_embeddings_1 = np.load("data/embedding/position_embeddings_1.npy")
    position_embeddings_2 = np.load("data/embedding/position_embeddings_2.npy")
    char_embeddings = np.load("data/embedding/char_embeddings.npy")
    tag_embeddings = np.load("data/embedding/tag_embeddings.npy")
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2, char_embeddings, tag_embeddings)

    print("training")
    config = ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)

    if not os.path.exists("model"):
        os.makedirs("model")
    filepath = "model/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model = build_model(embeddings)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True, callbacks=[checkpoint], validation_data=[x_test, y_test])

    print("testing")
    model.load_weights("model/weights.best.hdf5")
    scores = model.predict(x_test, verbose=False)
    predictions = scores.argmax(-1)
    meta = """
baseline:
POS tagging: yes
activation: relu
characted-level: yes
"""
    gen_answer_key(predictions, meta=meta)


if __name__ == "__main__":
    main()
