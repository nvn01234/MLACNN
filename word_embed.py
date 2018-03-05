import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os


def read_word_embeddings():
    word2vec = {
        "UNKNOWN": np.random.uniform(-0.25, 0.25, size=WORD_EMBED_SIZE),
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
        self.max_sequence_len = 0
        self.word2vec = word2vec
        self.unknown_words = set()

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

    def extract_entity_contexts(self):
        self.e1_context = []
        if self.e1pos > 0:
            self.e1_context.append(self.word_embed(self.tokens[self.e1pos - 1]))
        else:
            self.e1_context.append(self.e1)
        self.e1_context.append(self.e1)
        if self.e1pos < len(self.tokens) - 1:
            self.e1_context.append(self.word_embed(self.tokens[self.e1pos + 1]))
        else:
            self.e1_context.append(self.e1)

        self.e2_context = []
        if self.e2pos > 0:
            self.e2_context.append(self.word_embed(self.tokens[self.e2pos - 1]))
        else:
            self.e2_context.append(self.e2)
        self.e2_context.append(self.e2)
        if self.e2pos < len(self.tokens) - 1:
            self.e2_context.append(self.word_embed(self.tokens[self.e2pos + 1]))
        else:
            self.e2_context.append(self.e2)

    def extract_word_feature(self):
        self.words = []
        for i in range(SEQUENCE_LEN):
            if i < len(self.tokens):
                self.words.append(self.word_embed(self.tokens[i]))
            else:
                self.words.append(np.zeros(WORD_EMBED_SIZE))

    def feed(self, data):
        data = data.strip().split("\n")[0]
        data = data.strip().split("\t")[1][1:-1]

        self.data = []
        self.e1 = None
        self.e2 = None
        super(SemEvalParser, self).feed(data)

        self.tokens = word_tokenize(" ".join(self.data))
        self.max_sequence_len = max(self.max_sequence_len, len(self.tokens))
        self.find_entity_pos()

        self.tokens = [w[3:] if w == self.e1 or w == self.e2 else w for w in self.tokens]
        self.e1 = self.e1[3:]
        self.e2 = self.e2[3:]

        self.e1 = self.word_embed(self.e1)
        self.e2 = self.word_embed(self.e2)

        self.extract_entity_contexts()
        self.extract_word_feature()

    def word_embed(self, w):
        w = w.lower()
        if w in self.word2vec:
            return self.word2vec[w]
        elif "_" in w:
            w = w.split("_")
            temp = []
            for _w in w:
                if _w in self.word2vec:
                    temp.append(self.word2vec[_w])
                else:
                    self.unknown_words.add(_w)
                    temp.append(self.word2vec["UNKNOWN"])
            return np.average(temp, 0)
        else:
            self.unknown_words.add(w)
            return self.word2vec["UNKNOWN"]


def read_file(path, parser):
    words = []
    e1 = []
    e2 = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            parser.feed(record)
            words.append(parser.words)
            e1.append(parser.e1_context)
            e2.append(parser.e2_context)

    return words, e1, e2


def main():
    for folder in ["data", "data/train", "data/test"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print("read word embeddings")
    word2vec = read_word_embeddings()
    parser = SemEvalParser(word2vec)

    print("read train data")
    words_train, e1_train, e2_train = read_file("origin_data/TRAIN_FILE.TXT", parser)
    np.save("data/train/words.npy", words_train)
    np.save("data/train/e1.npy", e1_train)
    np.save("data/train/e2.npy", e2_train)

    print("read test data")
    words_test, e1_test, e2_test = read_file("origin_data/TEST_FILE_FULL.TXT", parser)
    np.save("data/test/words.npy", words_test)
    np.save("data/test/e1.npy", e1_test)
    np.save("data/test/e2.npy", e2_test)

    print("max_sequence_len: %d, unknown words: %d" % (parser.max_sequence_len, len(parser.unknown_words)))


if __name__ == "__main__":
    main()
