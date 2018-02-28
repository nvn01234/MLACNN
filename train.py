import os
import tensorflow as tf
import numpy as np
import time
from tensorflow import ConfigProto
from settings import *
from models import Model
from sklearn.metrics import f1_score
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_epoch(session, model, batch_iter, is_training=True, verbose=True):
    start_time = time.time()
    acc_count = 0
    step = 0  # len(all_data)
    y_true = []
    y_pred = []
    for batch in batch_iter:
        step += 1
        batch = (x for x in zip(*batch))
        sents, relations, e1, e2, dist1, dist2 = batch
        # sents is a list of np.ndarray, convert it to a single np.ndarray
        sents = np.vstack(sents)

        in_x, in_e1, in_e2, in_dist1, in_dist2, in_y = model.inputs
        feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1,
                     in_dist2: dist2, in_y: relations}
        y_true += list(relations)
        if is_training:
            _, _, acc, loss, pred = session.run([model.train_op, model.reg_op, model.acc, model.loss, model.predict],
                                                feed_dict=feed_dict)
            acc_count += acc
            if verbose and step % 10 == 0:
                logging.info("  step: %d acc: %.2f%% loss: %.2f time: %.2f" % (
                    step,
                    acc_count / (step * BATCH_SIZE) * 100,
                    loss,
                    time.time() - start_time
                ))
        else:
            acc, pred = session.run([model.acc, model.predict], feed_dict=feed_dict)
            acc_count += acc
        y_pred += list(pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc_count / (step * BATCH_SIZE), f1


def vectorize(paths):
    data = []
    for path in paths:
        d = np.load(path)
        if d.dtype == np.int64:
            d = d.astype(np.int32)
        elif d.dtype == np.float64:
            d = d.astype(np.float32)
        data.append(d)
    return data


def init():
    # Config log
    if LOG_FILE is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    elif SAVE_PATH is not None:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        logging.basicConfig(filename=LOG_FILE,
                            filemode='a', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    # Load data
    # vec = (sents_vec, relations, e1_vec, e2_vec, dist1, dist2)
    train_data = [
        X_WORDS_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_E1_TRAIN_PATH,
        X_E2_TRAIN_PATH,
        X_POS1_TRAIN_PATH,
        X_POS2_TRAIN_PATH,
    ]
    train_vec = vectorize(train_data)
    test_data = [
        X_WORDS_TEST_PATH,
        Y_TEST_PATH,
        X_E1_TEST_PATH,
        X_E2_TEST_PATH,
        X_POS1_TEST_PATH,
        X_POS2_TEST_PATH,
    ]
    test_vec = vectorize(test_data)

    return train_vec, test_vec


def batch_iter(data, shuffle=True):
    """
    Generates batches for the NN input feed.

    Returns a generator (yield) as the datasets are expected to be huge.
    """
    data = np.array(data)
    data_size = len(data)

    batches_per_epoch = data_size // BATCH_SIZE

    # logging.info("Generating batches.. Total # of batches %d" % batches_per_epoch)

    if shuffle:
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]
    else:
        shuffled_data = data
    for batch_num in range(batches_per_epoch):
        start_index = batch_num * BATCH_SIZE
        end_index = min((batch_num + 1) * BATCH_SIZE, data_size)
        yield shuffled_data[start_index:end_index]


def main():
    train_vec, test_vec = init()
    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                m_train = Model(is_training=True)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True):
                m_test = Model(is_training=False)

        sv = tf.train.Supervisor(logdir=SAVE_PATH,
                                 global_step=m_train.global_step)
        config = ConfigProto()
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as session:
            if TEST_ONLY:
                test_iter = batch_iter(list(zip(*test_vec)), shuffle=False)
                test_acc, test_f1 = run_epoch(session, m_test, test_iter, is_training=False)
                print("test acc: %.3f, test f1: %.3f" % (test_acc, test_f1))
            else:
                for epoch in range(NB_EPOCHS):
                    train_iter = batch_iter(list(zip(*train_vec)), shuffle=True)
                    test_iter = batch_iter(list(zip(*test_vec)), shuffle=False)
                    train_acc, train_f1 = run_epoch(session, m_train, train_iter, verbose=False)
                    test_acc, test_f1 = run_epoch(session, m_test, test_iter, is_training=False)
                    logging.info("Epoch: %d, Train: %.2f%%, Test: %.2f%%, Train f1: %.2f%%, Test f1: %.2f%%" %
                                 (epoch + 1, train_acc * 100, test_acc * 100, train_f1 * 100, test_f1 * 100))
                if SAVE_PATH is not None:
                    sv.saver.save(session, SAVE_PATH, global_step=sv.global_step)


if __name__ == "__main__":
    main()