import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os

def read_word_embeddings():
    word2vec = {
        "UNKNOWN": np.random.uniform(-0.01, 0.01, size=WORD_EMBED_SIZE),
    }
    with open(ORIGIN_WORD_EMBEDDINGS_PATH, "r", encoding="utf8") as f:
        for line in f:
            w, *values = line.strip().split()
            values = np.array(values, dtype='float32')
            word2vec[w] = values
    return word2vec


class SemEvalParser(HTMLParser):
    def __init__(self, word2vec):
        super(SemEvalParser, self).__init__()
        self.max_len = 0
        self.word2vec = word2vec
        self.num_unk = 0

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
        self.max_len = max(self.max_len, len(tokens))
        for i, w in enumerate(tokens):
            if self.e1 == w:
                self.e1pos = i
            if self.e2 == w:
                self.e2pos = i

        self.e1 = self.word_embed(self.e1[3:])
        self.e2 = self.word_embed(self.e2[3:])

        self.words = []
        self.pos1 = []
        self.pos2 = []
        for i in range(SEQUENCE_LEN):
            if i < len(tokens):
                token = tokens[i][3:] if i == self.e1pos or i == self.e2pos else tokens[i]
                self.words.append(self.word_embed(token))
            else:
                self.words.append(np.zeros(WORD_EMBED_SIZE))
            self.pos1.append(self.pos_embed(i - self.e1pos))
            self.pos2.append(self.pos_embed(i - self.e2pos))

    def pos_embed(self, d):
        if d < MIN_DISTANCE:
            d = MIN_DISTANCE
        elif d > MAX_DISTANCE:
            d = MAX_DISTANCE
        return d - MIN_DISTANCE

    def word_embed(self, w):
        w = w.strip().lower().split("_")
        temp = []
        for _w in w:
            if _w in self.word2vec:
                temp.append(self.word2vec[_w])
            else:
                self.num_unk += 1
                temp.append(self.word2vec["UNKNOWN"])
        return np.average(temp, 0)


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


def read_file(path, parser):
    x_words = []
    x_pos1 = []
    x_pos2 = []
    x_e1 = []
    x_e2 = []
    y = []
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

    return x_words, x_pos1, x_pos2, x_e1, x_e2, y


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    print("read word embeddings")
    word2vec = read_word_embeddings()
    parser = SemEvalParser(word2vec)

    print("read train data")
    x_words_train, x_pos1_train, x_pos2_train, x_e1_train, x_e2_train, y_train = read_file(TRAIN_FILE, parser)
    np.save(X_WORDS_TRAIN_PATH, x_words_train)
    np.save(X_E1_TRAIN_PATH, x_e1_train)
    np.save(X_E2_TRAIN_PATH, x_e2_train)
    np.save(X_POS1_TRAIN_PATH, x_pos1_train)
    np.save(X_POS2_TRAIN_PATH, x_pos2_train)
    np.save(Y_TRAIN_PATH, y_train)

    print("read test data")
    x_words_test, x_pos1_test, x_pos2_test, x_e1_test, x_e2_test, y_test = read_file(TEST_FILE, parser)
    np.save(X_WORDS_TEST_PATH, x_words_test)
    np.save(X_E1_TEST_PATH, x_e1_test)
    np.save(X_E2_TEST_PATH, x_e2_test)
    np.save(X_POS1_TEST_PATH, x_pos1_test)
    np.save(X_POS2_TEST_PATH, x_pos2_test)
    np.save(Y_TEST_PATH, y_test)

    print("maxlen: %d, num unknown: %d" % (parser.max_len, parser.num_unk))

if __name__ == "__main__":
    main()
