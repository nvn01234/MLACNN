import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os
from gensim.models import Word2Vec


def read_word_embeddings():
    word2vec = {
        "UNKNOWN": np.random.uniform(-0.01, 0.01, size=WORD_EMBED_SIZE),
    }
    with open("origin_data/glove.6B.300d.txt", "r", encoding="utf8") as f:
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
        self.pos1_total = []
        self.pos2_total = []

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

        self.e1 = self.e1[3:]
        self.e2 = self.e2[3:]
        tokens = [t[3:] if i == self.e1pos or i == self.e2pos else t for i, t in enumerate(tokens)]

        self.e1 = self.word_embed(self.e1)
        self.e2 = self.word_embed(self.e2)

        self.words = []
        for i in range(SEQUENCE_LEN):
            if i < len(tokens):
                self.words.append(self.word_embed(tokens[i]))
            else:
                self.words.append(np.zeros(WORD_EMBED_SIZE))

        self.pos1 = []
        self.pos2 = []
        for i in range(len(tokens)):
            self.pos1.append(self.pos_embed(i - self.e1pos))
            self.pos2.append(self.pos_embed(i - self.e2pos))
        print(self.pos1)
        self.pos1_total += self.pos1
        self.pos2_total += self.pos2

    def pos_embed(self, d):
        if d < MIN_DISTANCE:
            d = MIN_DISTANCE
        elif d > MAX_DISTANCE:
            d = MAX_DISTANCE
        return str(d)

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
    words = []
    pos1 = []
    pos2 = []
    e1 = []
    e2 = []
    labels = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            parser.feed(record)
            words.append(parser.words)
            pos1.append(parser.pos1)
            pos2.append(parser.pos2)
            e1.append(parser.e1)
            e2.append(parser.e2)
            labels.append(parser.label)

    return words, pos1, pos2, e1, e2, labels


def pretrain_pos2vec(pos1, pos2):
    pos2vec_1 = Word2Vec(pos1, POSITION_EMBED_SIZE, min_count=1)
    pos2vec_1.init_sims()
    pos2vec_1.wv.save_word2vec_format("data/embedding/position_embeddings_1.txt", binary=False)

    pos2vec_2 = Word2Vec(pos2, POSITION_EMBED_SIZE, min_count=1)
    pos2vec_2.init_sims()
    pos2vec_2.wv.save_word2vec_format("data/embedding/position_embeddings_1.txt", binary=False)

    return pos2vec_1.wv, pos2vec_2.wv


def embed_lookup(poss, pos2vec):
    new_poss = []
    for pos in poss:
        new_pos = []
        for i in range(SEQUENCE_LEN):
            if i < len(pos):
                new_pos.append(pos2vec.word_vec(pos[i]))
            else:
                new_pos.append(np.zeros(POSITION_EMBED_SIZE))
        new_poss.append(new_pos)
    return new_poss


def main():
    for folder in ["data", "data/train", "data/test", "data/embedding"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print("read word embeddings")
    word2vec = read_word_embeddings()
    parser = SemEvalParser(word2vec)

    print("read train data")
    words_train, pos1_train, pos2_train, e1_train, e2_train, labels_train = read_file("origin_data/TRAIN_FILE.TXT",
                                                                                      parser)
    np.save("data/train/words.npy", words_train)
    np.save("data/train/e1.npy", e1_train)
    np.save("data/train/e2.npy", e2_train)
    np.save("data/train/labels.npy", labels_train)

    print("read test data")
    words_test, pos1_test, pos2_test, e1_test, e2_test, labels_test = read_file("origin_data/TEST_FILE_FULL.TXT",
                                                                                parser)
    np.save("data/test/words.npy", words_test)
    np.save("data/test/e1.npy", e1_test)
    np.save("data/test/e2.npy", e2_test)
    np.save("data/test/labels.npy", labels_test)

    print("maxlen: %d, num unknown: %d" % (parser.max_len, parser.num_unk))

    print("pretrain position embedding")
    pos2vec_1, pos2vec_2 = pretrain_pos2vec(parser.pos1_total, parser.pos2_total)
    pos1_train = embed_lookup(pos1_train, pos2vec_1)
    pos2_train = embed_lookup(pos2_train, pos2vec_2)
    pos1_test = embed_lookup(pos1_test, pos2vec_1)
    pos2_test = embed_lookup(pos2_test, pos2vec_2)
    np.save("data/train/pos1.npy", pos1_train)
    np.save("data/train/pos2.npy", pos2_train)
    np.save("data/test/pos1.npy", pos1_test)
    np.save("data/test/pos2.npy", pos2_test)


if __name__ == "__main__":
    main()
