import numpy as np
from settings import *
from utils import make_dict


class Counter:
    def __init__(self):
        self.max_sequence_len = 0
        self.max_word_len = 0
        self.max_entity_len = 0
        self.vocab_word = set()
        self.vocab_char = set()

    def update(self, sentence):
        self.max_sequence_len = max(self.max_sequence_len, len(sentence))
        self.max_word_len = max(self.max_word_len, sentence.max_word_len)
        self.max_entity_len = max(self.max_entity_len, sentence.max_entity_len)
        self.vocab_word = self.vocab_word | sentence.vocab_word
        self.vocab_char = self.vocab_char | sentence.vocab_char

    def __str__(self):
        return "max_sequence_len = %d, max_word_len = %d, max_entity_len = %d, vocab_word = %d, vocab_char = %d" % (self.max_sequence_len, self.max_word_len, self.max_entity_len, len(self.vocab_word), len(self.vocab_char))


def dis2pos(dis):
    if dis < MIN_DISTANCE:
        dis = MIN_DISTANCE
    if dis > MAX_DISTANCE:
        dis = MAX_DISTANCE
    return dis - MIN_DISTANCE + 1


class Sentence:
    def __init__(self, e1start, e1end, e2start, e2end, words):
        self.e1start = e1start
        self.e1end = e1end
        self.e2start = e2start
        self.e2end = e2end
        self.words = words

        self.max_entity_len = max(self.e1end - self.e1start + 1, self.e2end - self.e2start + 1)

        self.vocab_word = set()
        self.vocab_char = set()
        self.max_word_len = 0
        self.positions_1 = []
        self.positions_2 = []
        for i, w in enumerate(self.words):
            self.vocab_word.add(w)
            self.max_word_len = max(self.max_word_len, len(words))
            self.positions_1.append(dis2pos(i - e1start))
            self.positions_2.append(dis2pos(i - e2start))
            for c in w:
                self.vocab_char.add(c)

        self.words_encoded = np.zeros(SEQUENCE_LEN, dtype='int32')
        self.chars_encoded = np.zeros([SEQUENCE_LEN, WORD_LEN], dtype='int32')
        self.e1_context = None
        self.e2_context = None

    def __len__(self):
        return len(self.words)

    def generate_features(self, encoder):
        for i in range(min(len(self.words), SEQUENCE_LEN)):
            w = self.words[i]
            self.words_encoded[i] = encoder.word_vec(w)
            for j in range(min(len(w), WORD_LEN)):
                c = w[j]
                self.chars_encoded[i, j] = encoder.char_vec(c)

        self.e1_context = self.entity_context(self.e1start, encoder)
        self.e2_context = self.entity_context(self.e2start, encoder)

        return self.words_encoded, self.chars_encoded, self.positions_1, self.positions_2, self.e1_context, self.e2_context

    def entity_context(self, pos, encoder):
        prev_pos = min(0, pos - 1)
        next_pos = max(len(self.words) - 1, pos + 1)
        context = [
            encoder.word_vec(self.words[prev_pos]),
            encoder.word_vec(self.words[pos]),
            encoder.word_vec(self.words[next_pos]),
        ]
        return context


def read_file(path, counter):
    sentences = []
    y = []
    with open(path, "r", encoding="utf8") as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        r, e1start, e1end, e2start, e2end, *words = line.split()
        y.append(int(r))
        s = Sentence(int(e1start), int(e1end), int(e2start), int(e2end), words)
        counter.update(s)
        sentences.append(s)
    return sentences, y


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


class Encoder:
    def __init__(self, word2idx, char2idx):
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.unknown_words = set()
        self.unknown_chars = set()

    def word_vec(self, w):
        if w in self.word2idx:
            return self.word2idx[w]
        else:
            self.unknown_words.add(w)
            return self.word2idx["UNKNOWN"]

    def char_vec(self, c):
        if c in self.char2idx:
            return self.char2idx[c]
        else:
            self.unknown_chars.add(c)
            return self.char2idx["UNKNOWN"]

    def __str__(self):
        return "unknown_words: %d, unknown_chars: %d" % (len(self.unknown_words), len(self.unknown_chars))


def read_char_embeddings(vocab):
    char2idx = {
        "PADDING": 0,
        "UNKNOWN": 1,
    }
    char_embeddings = [
        np.zeros(CHAR_EMBED_SIZE),
        np.random.uniform(-0.25, 0.25, CHAR_EMBED_SIZE),
    ]
    with open("origin_data/char-embeddings.txt", "r", encoding="utf8") as f:
        for line in f:
            c, *values = line.strip().split()
            if len(values) < CHAR_EMBED_SIZE:
                values = [c] + values
                c = " "
            if c in vocab:
                char2idx[c] = len(char2idx)
                values = np.array(values, dtype='float32')
                char_embeddings.append(values)
    np.save("data/embedding/char_embeddings.npy", char_embeddings)
    return char2idx


def numpy_save_many(_dict_):
    for k, data in _dict_:
        name, folder = k.split("_")
        np.save("%s/%s.npy" % (folder, name), data)


def main():
    counter = Counter()

    print("read train file")
    sentences_train, y_train = read_file("origin_data/train.cln", counter)

    print("read test file")
    sentences_test, y_test = read_file("origin_data/test.cln", counter)

    print(counter)

    print("read word embeddings")
    word2idx = read_word_embeddings(counter.vocab_word)

    print("read char embeddings")
    char2idx = read_char_embeddings(counter.vocab_char)

    encoder = Encoder(word2idx, char2idx)

    print("saving train data")
    words_train, chars_train, pos1_train, pos2_train, e1_train, e2_train = zip(*[s.generate_features(encoder) for s in sentences_train])
    data_train = make_dict(words_train, chars_train, pos1_train, pos2_train, e1_train, e2_train, y_train)
    numpy_save_many(data_train)

    print("saving test data")
    words_test, chars_test, pos1_test, pos2_test, e1_test, e2_test = zip(*[s.generate_features(encoder) for s in sentences_test])
    data_test = make_dict(words_test, chars_test, pos1_test, pos2_test, e1_test, e2_test, y_test)
    numpy_save_many(data_test)


if __name__ == "__main__":
    main()
