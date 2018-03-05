import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os
from keras.preprocessing.sequence import pad_sequences


def read_word_embeddings(vocab):
    word2idx = {
        "PADDING": 0,
        "UNKNOWN": 1,
    }
    word_embeddings = [
        np.zeros(WORD_EMBED_SIZE),
        np.random.uniform(-0.25, 0.25, WORD_EMBED_SIZE)
    ]
    with open("origin_data/glove.6B.300d.txt", "r", encoding="utf8") as f:
        for line in f:
            w, *values = line.strip().split()
            if w in vocab:
                values = np.array(values, dtype='float32')
                word2idx[w] = len(word2idx)
                word_embeddings.append(values)
    np.save("data/embedding/word_embeddings.npy", word_embeddings)
    return word2idx


class SemEvalParser(HTMLParser):
    def __init__(self):
        super(SemEvalParser, self).__init__()
        self.max_sequence_len = 0
        self.vocab = set()

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

        self.data = []
        self.e1 = None
        self.e2 = None
        super(SemEvalParser, self).feed(data)

        words = word_tokenize(" ".join(self.data))

        self.max_sequence_len = max(self.max_sequence_len, len(words))

        self.e1pos = 0
        self.e2pos = 0
        self.words = []
        for i, w in enumerate(words):
            if w == self.e1:
                self.e1pos = i
            if w == self.e2:
                self.e2pos = i
            if w == self.e1 or w == self.e2:
                w = w[3:]
            self.words.append(w)

        self.vocab = set(list(self.vocab) + self.words)


def read_file(path, parser):
    words = []
    e1pos = []
    e2pos = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            parser.feed(record)
            words.append(parser.words)
            e1pos.append(parser.e1)
            e2pos.append(parser.e2)

    return words, e1pos, e2pos

class Word2Vec:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.unknown_words = set()

    def word_vec(self, w):
        if w in self.word2idx:
            return self.word2idx[w]
        else:
            self.unknown_words.add(w)
            return self.word2idx["UNKNOWN"]

def extract_features(words, e1pos, e2pos, word2vec):
    words = [[word2vec.word_vec(w) for w in sen] for sen in words]
    e1context = entity_context(words, e1pos)
    e2context = entity_context(words, e2pos)
    words = pad_sequences(words, maxlen=SEQUENCE_LEN, padding="post", truncating="post", value=0)
    return words, e1context, e2context


def entity_context(words, pos):
    context = []
    if pos > 0:
        context.append(words[pos - 1])
    else:
        context.append(words[pos])
    context.append(words[pos])
    if pos < len(words) - 1:
        context.append(words[pos + 1])
    else:
        context.append(words[pos])
    return context

def main():
    for folder in ["data", "data/train", "data/test", "data/embedding"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    parser = SemEvalParser()

    print("read train data")
    words_train, e1pos_train, e2pos_train = read_file("origin_data/TRAIN_FILE.TXT", parser)

    print("read test data")
    words_test, e1pos_test, e2pos_test = read_file("origin_data/TEST_FILE_FULL.TXT", parser)

    print("max_sequence_len: %d, vocab size: %d" % (parser.max_sequence_len, len(parser.vocab)))

    print("read word embeddings")
    word2idx = read_word_embeddings(parser.vocab)
    word2vec = Word2Vec(word2idx)

    print("extract features")
    words_train, e1_train, e2_train = extract_features(words_train, e1pos_train, e2pos_train, word2vec)
    words_test, e1_test, e2_test = extract_features(words_test, e1pos_test, e2pos_test, word2vec)

    print("unknown words: %d" % len(word2vec.unknown_words))

    print("saving data")
    np.save("data/train/words.npy", words_train)
    np.save("data/train/e1.npy", e1_train)
    np.save("data/train/e2.npy", e2_train)
    np.save("data/test/words.npy", words_test)
    np.save("data/test/e1.npy", e1_test)
    np.save("data/test/e2.npy", e2_test)


if __name__ == "__main__":
    main()
