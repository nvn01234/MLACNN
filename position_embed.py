import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os

"""
Do not using word2vec to pretrain position embedding
"""

class SemEvalParser(HTMLParser):
    def __init__(self):
        super(SemEvalParser, self).__init__()
        self.data = []
        self.e1 = None
        self.e2 = None

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

    def find_entity_pos(self):
        self.e1pos = 0
        self.e2pos = 0
        for i, w in enumerate(self.tokens):
            if self.e1 == w:
                self.e1pos = i
            if self.e2 == w:
                self.e2pos = i

    def feed(self, data):
        data = data.strip().split("\n")[0]
        data = data.strip().split("\t")[1][1:-1]
        super(SemEvalParser, self).feed(data)

        self.tokens = word_tokenize(" ".join(self.data))

        self.find_entity_pos()

        dis1 = []
        dis2 = []
        for i in range(SEQUENCE_LEN):
            e1dis = reduce_dis(i - self.e1pos)
            e2dis = reduce_dis(i - self.e2pos)
            dis1.append(e1dis)
            dis2.append(e2dis)
        return dis1, dis2


def reduce_dis(dis):
    if dis < MIN_DISTANCE:
        dis = MIN_DISTANCE
    elif dis > MAX_DISTANCE:
        dis = MAX_DISTANCE
    return dis - MIN_DISTANCE + 1


def read_file(path):
    distances_1 = []
    distances_2 = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            parser = SemEvalParser()
            dis1, dis2 = parser.feed(record)
            distances_1.append(dis1)
            distances_2.append(dis2)
    return distances_1, distances_2


def main():
    for folder in ["data", "data/train", "data/test"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print("load train file")
    distances_1_train, distances_2_train = read_file("origin_data/TRAIN_FILE.TXT")

    print("load test file")
    distances_1_test, distances_2_test = read_file("origin_data/TEST_FILE_FULL.TXT")

    np.save("data/train/pos1.npy", distances_1_train)
    np.save("data/train/pos2.npy", distances_2_train)
    np.save("data/test/pos1.npy", distances_1_test)
    np.save("data/test/pos2.npy", distances_2_test)


if __name__ == "__main__":
    main()
