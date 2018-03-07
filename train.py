import numpy as np
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
    model = build_model(embeddings)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True)

    print("testing")
    scores = model.predict(x_test, verbose=False)
    predictions = scores.argmax(-1)
    meta = """
baseline
position embedding: pretrain, trainable, size = 5
pos tag: pretrain, trainable, size = 10
char embedding: 50 filters, window size: [3]
attention input: Yes, concat, hidden layer: 50
word-level convo: 200 filter, window size: [3]
"""
    gen_answer_key(predictions, meta=meta)


if __name__ == "__main__":
    main()
