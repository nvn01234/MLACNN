import numpy as np
from gensim.models import Word2Vec
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import traceback
import os


def make_dict(*expr):
    (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
    begin = text.find('make_dict(') + len('make_dict(')
    end = text.find(')', begin)
    text = [name.strip() for name in text[begin:end].split(',')]
    return dict(zip(text, expr))


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

    def feed(self, data):
        data = data.strip().split("\n")[0]
        data = data.strip().split("\t")[1][1:-1]
        super(SemEvalParser, self).feed(data)

        tokens = word_tokenize(" ".join(self.data))

        e1pos = 0
        e2pos = 0
        for i, w in enumerate(tokens):
            if self.e1 == w:
                e1pos = i
            if self.e2 == w:
                e2pos = i

        dis1 = []
        dis2 = []
        for i in range(len(tokens)):
            e1dis = reduce_dis(i - e1pos)
            e2dis = reduce_dis(i - e2pos)
            dis1.append(e1dis)
            dis2.append(e2dis)
        return dis1, dis2


def reduce_dis(dis):
    if dis < MIN_DISTANCE:
        dis = MIN_DISTANCE
    elif dis > MAX_DISTANCE:
        dis = MAX_DISTANCE
    return str(dis)


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


def to_vec(distances, dis2vec):
    new_distances = []
    for dis in distances:
        new_dis = []
        for i in range(SEQUENCE_LEN):
            if i < len(dis):
                new_dis.append(dis2vec.word_vec(dis[i]))
            else:
                new_dis.append(np.zeros(POSITION_EMBED_SIZE))
        new_distances.append(new_dis)
    return new_distances



def main():
    for folder in ["data", "data/train", "data/test", "data/embedding"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print("load train file")
    distances_1_train, distances_2_train = read_file("origin_data/TRAIN_FILE.TXT")

    print("load test file")
    distances_1_test, distances_2_test = read_file("origin_data/TEST_FILE_FULL.TXT")

    print("pretrain position embeddings")
    distances_1 = distances_1_train + distances_1_test
    dis2vec_1 = Word2Vec(distances_1, size=POSITION_EMBED_SIZE, min_count=1)
    dis2vec_1.init_sims(replace=True)  # remove syn1, replace syn0
    dis2vec_1 = dis2vec_1.wv
    dis2vec_1.save_word2vec_format("data/embedding/position_embeddings_1.txt", binary=False)

    distances_2 = distances_2_train + distances_2_test
    dis2vec_2 = Word2Vec(distances_2, size=POSITION_EMBED_SIZE, min_count=1)
    dis2vec_2.init_sims(replace=True)  # remove syn1, replace syn0
    dis2vec_2 = dis2vec_2.wv
    dis2vec_2.save_word2vec_format("data/embedding/position_embeddings_2.txt", binary=False)

    pos1_train = to_vec(distances_1_train, dis2vec_1)
    pos2_train = to_vec(distances_2_train, dis2vec_2)
    pos1_test = to_vec(distances_1_test, dis2vec_1)
    pos2_test = to_vec(distances_2_test, dis2vec_2)
    np.save("data/train/pos1.npy", pos1_train)
    np.save("data/train/pos2.npy", pos2_train)
    np.save("data/test/pos1.npy", pos1_test)
    np.save("data/test/pos2.npy", pos2_test)


if __name__ == "__main__":
    main()
