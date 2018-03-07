import numpy as np
from tensorflow import ConfigProto, Session
from keras import backend as K
from models import build_model
from settings import *
from test import gen_answer_key


def main():
    print("load train data")
    x_train = [np.load("data/train/%s.npy" % name) for name in ["words", "pos1", "pos2", "e1", "e2", "e1context", "e2context"]]
    y_train = np.load("data/train/y.npy")

    print("load test data")
    x_test = [np.load("data/test/%s.npy" % name) for name in ["words", "pos1", "pos2", "e1", "e2", "e1context", "e2context"]]

    print("training")
    config = ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)
    model = build_model()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True)

    print("testing")
    scores = model.predict(x_test, verbose=False)
    predictions = scores.argmax(-1)
    meta = """
position embedding: trainable
pos tag: False
char embedding: False
attention input: 100
"""
    gen_answer_key(predictions, meta=meta)


if __name__ == "__main__":
    main()
