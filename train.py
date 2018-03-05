import numpy as np
from tensorflow import ConfigProto, Session
from keras import backend as K
from models import build_model
from settings import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



def main():
    print("load train data")
    words_train = np.load("data/train/words.npy")
    pos1_train = np.load("data/train/pos1.npy")
    pos2_train = np.load("data/train/pos2.npy")
    e1_train = np.load("data/train/e1.npy")
    e2_train = np.load("data/train/e2.npy")
    chars_train = np.load("data/train/chars.npy")
    labels_train = np.load("data/train/labels.npy")
    x_train = [words_train, pos1_train, pos2_train, e1_train, e2_train, chars_train]

    print("load test data")
    words_test = np.load("data/test/words.npy")
    pos1_test = np.load("data/test/pos1.npy")
    pos2_test = np.load("data/test/pos2.npy")
    e1_test = np.load("data/test/e1.npy")
    e2_test = np.load("data/test/e2.npy")
    chars_test = np.load("data/test/chars.npy")
    labels_test = np.load("data/test/labels.npy")
    x_test = [words_test, pos1_test, pos2_test, e1_test, e2_test, chars_test]

    print("training")
    config = ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)
    model = build_model()
    model.fit(x_train, labels_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True)

    print("testing")
    scores = model.predict(x_test,verbose=False)
    predictions = scores.argmax(-1)
    accuracy = accuracy_score(labels_test, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(labels_test, predictions, average="macro", warn_for=())
    print("accuracy = %.4f%%, precision = %.4f%%, recall = %.4f%%, f1 = %.4f%%" % (accuracy, precision, recall, f1))
    np.save("log/y_pred_2.npy", predictions)



if __name__ == "__main__":
    main()
