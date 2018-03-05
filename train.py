import numpy as np
from tensorflow import ConfigProto, Session
from keras import backend as K
from models import build_model
from settings import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def main():
    print("load train data")
    x_train = [np.load("data/train/%s.npy" % name) for name in ["words", "chars", "pos1", "pos2", "e1", "e2"]]
    y_train = np.load("data/train/y.npy")

    print("load test data")
    x_test = [np.load("data/test/%s.npy" % name) for name in ["words", "chars", "pos1", "pos2", "e1", "e2"]]
    y_test = np.load("data/test/y.npy")

    print("training")
    config = ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)
    model = build_model()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=True)

    print("testing")
    scores = model.predict(x_test,verbose=False)
    predictions = scores.argmax(-1)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average="macro", warn_for=())
    print("accuracy = %.4f%%, precision = %.4f%%, recall = %.4f%%, f1 = %.4f%%" % (accuracy, precision, recall, f1))
    np.save("log/y_pred_3.npy", predictions)


if __name__ == "__main__":
    main()
