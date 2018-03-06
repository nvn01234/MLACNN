import numpy as np
from settings import *
from utils import make_dict
from os import makedirs
from gensim.models import Word2Vec
from nltk import pos_tag


class Counter:
    def __init__(self):
        self.max_sequence_len = 0
        self.max_word_len = 0
        self.max_entity_len = 0
        self.vocab_word = set()
        self.vocab_char = set()
        self.distances_1 = []
        self.distances_2 = []
        self.tags = []

    def update(self, sentence):
        self.max_sequence_len = max(self.max_sequence_len, len(sentence))
        self.max_word_len = max(self.max_word_len, sentence.max_word_len)
        self.max_entity_len = max(self.max_entity_len, sentence.max_entity_len)
        self.vocab_word = self.vocab_word | sentence.vocab_word
        self.vocab_char = self.vocab_char | sentence.vocab_char
        self.distances_1.append(sentence.distances_1)
        self.distances_2.append(sentence.distances_2)
        self.tags.append(sentence.tags)

    def __str__(self):
        return "max_sequence_len = %d, max_word_len = %d, max_entity_len = %d, vocab_word = %d, vocab_char = %d" % (self.max_sequence_len, self.max_word_len, self.max_entity_len, len(self.vocab_word), len(self.vocab_char))


def relative_distance(i, e_start, e_end):
    if i < e_start:
        dis = i - e_start
    elif e_start <= i <= e_end:
        dis = 0
    else:
        dis = i - e_end

    if dis < MIN_DISTANCE:
        dis = MIN_DISTANCE
    if dis > MAX_DISTANCE:
        dis = MAX_DISTANCE
    return str(dis)


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
        tags = pos_tag(words, 'universal')
        self.tags = []
        for i, (w, t) in enumerate(tags):
            self.vocab_word.add(w)
            self.tags.append(t)
            self.max_word_len = max(self.max_word_len, len(w))
            for c in w:
                self.vocab_char.add(c)

        self.distances_1 = []
        self.distances_2 = []
        for i in range(len(words)):
            self.distances_1.append(relative_distance(i, e1start, e1end))
            self.distances_2.append(relative_distance(i, e2start, e2end))

        self.positions_1 = []
        self.positions_2 = []
        self.tags_encoded = np.zeros(SEQUENCE_LEN, dtype='int32')
        self.words_encoded = np.zeros(SEQUENCE_LEN, dtype='int32')
        self.chars_encoded = np.zeros([SEQUENCE_LEN, WORD_LEN], dtype='int32')
        self.e1_context = None
        self.e2_context = None

    def __len__(self):
        return len(self.words)

    def generate_features(self, encoder):
        for i in range(min(len(self.words), SEQUENCE_LEN)):
            w = self.words[i]
            t = self.tags[i]
            self.words_encoded[i] = encoder.word_vec(w)
            self.tags_encoded[i] = encoder.tag_vec(t)
            for j in range(min(len(w), WORD_LEN)):
                c = w[j]
                self.chars_encoded[i, j] = encoder.char_vec(c)

        for i in range(SEQUENCE_LEN):
            self.positions_1.append(encoder.dis1_vec(i, self.e1start, self.e1end))
            self.positions_2.append(encoder.dis2_vec(i, self.e2start, self.e2end))

        self.e1_context = self.entity_context(self.e1start, encoder)
        self.e2_context = self.entity_context(self.e2start, encoder)

        return self.words_encoded, self.chars_encoded, self.positions_1, self.positions_2, self.e1_context, self.e2_context, self.tags_encoded

    def entity_context(self, estart, encoder):
        start = estart - 1
        end = estart + ENTITY_LEN
        context = []
        for i in range(start, end + 1):
            if i < 0:
                context.append(encoder.word_vec(self.words[0]))
            elif i > len(self.words) - 1:
                context.append(encoder.word_vec(self.words[-1]))
            else:
                context.append(encoder.word_vec(self.words[i]))
        return context


def read_file(path, counter):
    sentences = []
    y = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            r, e1start, e1end, e2start, e2end, *words = line.strip().split()
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
        np.random.normal(0, 0.1, WORD_EMBED_SIZE)
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
    def __init__(self, word2idx, char2idx, dis2idx_1, dis2idx_2, tag2idx):
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.dis2idx_1 = dis2idx_1
        self.dis2idx_2 = dis2idx_2
        self.tag2idx = tag2idx
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

    def dis1_vec(self, i, e_start, e_end):
        d = relative_distance(i, e_start, e_end)
        return self.dis2idx_1[d]

    def dis2_vec(self, i, e_start, e_end):
        d = relative_distance(i, e_start, e_end)
        return self.dis2idx_2[d]

    def tag_vec(self, t):
        return self.tag2idx[t]

    def __str__(self):
        return "unknown_words: %d, unknown_chars: %d" % (len(self.unknown_words), len(self.unknown_chars))


def read_char_embeddings(vocab):
    char2idx = {
        "PADDING": 0,
        "UNKNOWN": 1,
    }
    char_embeddings = [
        np.zeros(CHAR_EMBED_SIZE),
        np.random.normal(0, 0.1, CHAR_EMBED_SIZE),
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
    for k, data in _dict_.items():
        name, folder = k.split("_")
        np.save("data/%s/%s.npy" % (folder, name), data)


def pretrain_embedding(data, size, padding=False):
    model = Word2Vec(data, size=size, min_count=1)
    model.init_sims(replace=True)
    index = {}
    embeddings = []
    if padding:
        index["PADDING"] = len(index)
        embeddings.append(np.zeros(size))
    for d in model.wv.index2word:
        index[d] = len(index)
        embeddings.append(model.wv.word_vec(d))
    return index, embeddings


def main():
    for folder in ["data/train", "data/test", "data/embedding"]:
        makedirs(folder, exist_ok=True)

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

    print("pretrain position embeddings")
    dis2idx_1, position_embeddings_1 = pretrain_embedding(counter.distances_1, POSITION_EMBED_SIZE)
    dis2idx_2, position_embeddings_2 = pretrain_embedding(counter.distances_2, POSITION_EMBED_SIZE)
    np.save("data/embedding/position_embeddings_1.npy", position_embeddings_1)
    np.save("data/embedding/position_embeddings_2.npy", position_embeddings_2)

    print("pretrain pos_tag embeddings")
    tag2idx, tag_embeddings = pretrain_embedding(counter.tags, TAG_EMBED_SIZE, True)
    np.save("data/embedding/tag_embeddings.npy", tag_embeddings)

    encoder = Encoder(word2idx, char2idx, dis2idx_1, dis2idx_2, tag2idx)

    print("saving train data")
    words_train, chars_train, pos1_train, pos2_train, e1_train, e2_train, tags_train = zip(*[s.generate_features(encoder) for s in sentences_train])
    data_train = make_dict(words_train, chars_train, pos1_train, pos2_train, e1_train, e2_train, tags_train, y_train)
    numpy_save_many(data_train)

    print("saving test data")
    words_test, chars_test, pos1_test, pos2_test, e1_test, e2_test, tags_test = zip(*[s.generate_features(encoder) for s in sentences_test])
    data_test = make_dict(words_test, chars_test, pos1_test, pos2_test, e1_test, e2_test, y_test)
    numpy_save_many(data_test)

    print(encoder)


if __name__ == "__main__":
    main()
