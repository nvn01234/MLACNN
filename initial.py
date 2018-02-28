import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os


def read_word_embeddings(vocab):
    word2idx = {
        "PADDING": 0,
        "UNKNOWN": 1,
    }
    word_embeddings = [
        np.zeros(WORD_EMBED_SIZE),
        np.random.normal(size=WORD_EMBED_SIZE),
    ]
    with open(ORIGIN_WORD_EMBEDDINGS_PATH, "r", encoding="utf8") as f:
        for line in f:
            w, *values = line.strip().split()
            if w in vocab:
                values = np.array(values, dtype='float64')
                word2idx[w] = len(word2idx)
                word_embeddings.append(values)
    np.save(WORD_EMBEDDINGS_PATH, word_embeddings)
    return word2idx


def init_position_embeddings():
    dis2idx = {"PADDING": 0, "LOWER_MIN": 1, "GREATER_MAX": 2}
    for dis in range(MIN_DISTANCE, MAX_DISTANCE + 1):
        dis2idx[dis] = len(dis2idx)
    return dis2idx


class SemEvalParser(HTMLParser):
    def __init__(self, dis2idx):
        super(SemEvalParser, self).__init__()
        self.vocab = set()
        self.dis2idx = dis2idx

    def handle_starttag(self, tag, attrs):
        super(SemEvalParser, self).handle_starttag(tag, attrs)
        setattr(self, tag, True)

    def handle_data(self, data):
        super(SemEvalParser, self).handle_data(data)
        data = data.strip()
        if self.e1 is True:
            data = "e1_" + data.replace(" ", "_")
            self.e1 = data
        elif self.e2 is True:
            data = "e2_" + data.replace(" ", "_")
            self.e2 = data
        self.data.append(data)

    def feed(self, data):
        data, label, _ = data.strip().split("\n")
        data = data.strip().split("\t")[1][1:-1]
        self.label = label2idx(label)

        self.data = []
        self.e1 = None
        self.e2 = None
        super(SemEvalParser, self).feed(data)

        self.e1pos = 0
        self.e2pos = 0
        tokens = word_tokenize(" ".join(self.data))
        for i, w in enumerate(tokens):
            if self.e1 == w:
                self.e1pos = i
            if self.e2 == w:
                self.e2pos = i

        self.e1 = self.e1[3:]
        self.e2 = self.e2[3:]

        self.words = []
        self.pos1 = []
        self.pos2 = []
        for i in range(SEQUENCE_LEN):
            if i < len(tokens):
                token = tokens[i][3:] if i == self.e1pos or i == self.e2pos else tokens[i]
                self.words.append(token)
                self.pos1.append(self.pos_embed(i - self.e1pos))
                self.pos2.append(self.pos_embed(i - self.e2pos))
                self.vocab.add(token)
            else:
                self.words.append("PADDING")
                self.pos1.append(self.dis2idx["PADDING"])
                self.pos2.append(self.dis2idx["PADDING"])

    def pos_embed(self, d):
        if d < MIN_DISTANCE:
            return self.dis2idx["LOWER_MIN"]
        elif d > MAX_DISTANCE:
            return self.dis2idx["GREATER_MAX"]
        else:
            return self.dis2idx[d]


def label2idx(label):
    labels_mapping = {'Other': 0,
                      'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                      'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                      'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                      'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                      'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                      'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                      'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                      'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                      'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
    return labels_mapping[label] if label in labels_mapping else labels_mapping["Other"]


def read_file(path, dis2idx):
    x_words = []
    x_pos1 = []
    x_pos2 = []
    x_e1 = []
    x_e2 = []
    y = []
    parser = SemEvalParser(dis2idx)
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            parser.feed(record)
            x_words.append(parser.words)
            x_pos1.append(parser.pos1)
            x_pos2.append(parser.pos2)
            x_e1.append(parser.e1)
            x_e2.append(parser.e2)
            y.append(parser.label)

    return x_words, x_pos1, x_pos2, x_e1, x_e2, y, parser.vocab


def deep_map(data, word2idx):
    if isinstance(data, list):
        data = [deep_map(d, word2idx) for d in data]
    elif isinstance(data, str):
        data = data.split("_")
        data = [word2idx[d] if d in word2idx else word2idx["UNKNOWN"] for d in data]
        data = np.average(data, 0)
        data = np.array(data, dtype='int64')
    return data


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    dis2idx = init_position_embeddings()

    print("read train data")
    x_words_train, x_pos1_train, x_pos2_train, x_e1_train, x_e2_train, y_train, vocab_train = read_file(TRAIN_FILE, dis2idx)
    np.save(X_POS1_TRAIN_PATH, x_pos1_train)
    np.save(X_POS2_TRAIN_PATH, x_pos2_train)
    np.save(Y_TRAIN_PATH, y_train)

    print("read test data")
    x_words_test, x_pos1_test, x_pos2_test, x_e1_test, x_e2_test, y_test, vocab_test = read_file(TEST_FILE, dis2idx)
    np.save(X_POS1_TEST_PATH, x_pos1_test)
    np.save(X_POS2_TEST_PATH, x_pos2_test)
    np.save(Y_TEST_PATH, y_test)

    print("read word embeddings")
    vocab = set(list(vocab_train) + list(vocab_test))
    word2idx = read_word_embeddings(vocab)
    x_words_train = deep_map(x_words_train, word2idx)
    x_e1_train = deep_map(x_e1_train, word2idx)
    x_e2_train = deep_map(x_e2_train, word2idx)
    x_words_test = deep_map(x_words_test, word2idx)
    x_e1_test = deep_map(x_e1_test, word2idx)
    x_e2_test = deep_map(x_e2_test, word2idx)
    np.save(X_WORDS_TRAIN_PATH, x_words_train)
    np.save(X_E1_TRAIN_PATH, x_e1_train)
    np.save(X_E2_TRAIN_PATH, x_e2_train)
    np.save(X_WORDS_TEST_PATH, x_words_test)
    np.save(X_E1_TEST_PATH, x_e1_test)
    np.save(X_E2_TEST_PATH, x_e2_test)


if __name__ == "__main__":
    main()
