import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize


def read_word_embeddings():
    print("read word embeddings")
    word2idx = {
        "PADDING": 0,
        "UNKNOWN": 1,
    }
    word_embeddings = [
        np.zeros(WORD_EMBED_SIZE),
        np.random.normal(size=WORD_EMBED_SIZE),
    ]
    with open("origin_data/glove.6B.300d.txt", "r", encoding="utf8") as f:
        for line in f:
            w, *values = line.strip().split()
            values = np.array(values, dtype='float64')
            word2idx[w] = len(word2idx)
            word_embeddings.append(values)
    np.save("data/word_embeddings.npy", word_embeddings)
    return word2idx


def init_position_embeddings():
    print("init position embeddings")
    dis2idx = {"PADDING": 0, "LOWER_MIN": 1, "GREATER_MAX": 2}
    for dis in range(MIN_DISTANCE, MAX_DISTANCE + 1):
        dis2idx[dis] = len(dis2idx)
    return dis2idx


class SemEvalParser(HTMLParser):
    def __init__(self, word2idx, dis2idx):
        super(SemEvalParser, self).__init__()
        self.data = []

        self.e1 = None
        self.e2 = None
        self.e1pos = 0
        self.e2pos = 0

        self.words = []
        self.pos1 = []
        self.pos2 = []

        self.word2idx = word2idx
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
        super(SemEvalParser, self).feed(data)

        tokens = word_tokenize(" ".join(self.data))
        for i, w in enumerate(tokens):
            if self.e1 == w:
                self.e1pos = i
            if self.e2 == w:
                self.e2pos = i

        self.e1 = self.word_embed(self.e1[3:])
        self.e2 = self.word_embed(self.e2[3:])
        tokens = [t[:3] if t.startswith("e1_") or t.startswith("e2_") else t for t in tokens]

        for i in range(SEQUENCE_LEN):
            if i < len(tokens):
                self.words.append(self.word_embed(tokens[i]))
                self.pos1.append(self.pos_embed(i - self.e1pos))
                self.pos2.append(self.pos_embed(i - self.e2pos))
            else:
                self.words.append(self.word2idx["PADDING"])
                self.pos1.append(self.dis2idx["PADDING"])
                self.pos2.append(self.dis2idx["PADDING"])

    def word_embed(self, w):
        return self.word2idx[w] if w in self.word2idx else self.word2idx["UNKNOWN"]

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


def read_file(_set, path, word2idx, dis2idx):
    print("read %s data" % _set)
    x_words = []
    x_pos1 = []
    x_pos2 = []
    x_e1 = []
    x_e2 = []
    y = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            text, label, _ = record.strip().split("\n")

            text = text.strip().split("\t")[1][1:-1]
            label = label2idx(label)
            parser = SemEvalParser(word2idx, dis2idx)
            parser.feed(text)

            x_words.append(parser.words)
            x_pos1.append(parser.pos1)
            x_pos2.append(parser.pos2)
            x_e1.append(parser.e1)
            x_e2.append(parser.e2)
            y.append(label)

    np.save("data/x_words_%s.npy" % _set, x_words)
    np.save("data/x_pos1_%s.npy" % _set, x_pos1)
    np.save("data/x_pos2_%s.npy" % _set, x_pos2)
    np.save("data/x_e1_%s.npy" % _set, x_e1)
    np.save("data/x_e2_%s.npy" % _set, x_e2)
    np.save("data/y_%s.npy" % _set, y)


def main():
    word2idx = read_word_embeddings()
    dis2idx = init_position_embeddings()
    read_file("train", "origin_data/TRAIN_FILE.TXT", word2idx, dis2idx)
    read_file("test", "origin_data/TEST_FILE_FULL.TXT", word2idx, dis2idx)


if __name__ == "__main__":
    main()
