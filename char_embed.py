import numpy as np
from settings import *
from html.parser import HTMLParser
from nltk import word_tokenize
import os


def read_char_embeddings():
    char2vec = {
        "UNKNOWN": np.random.uniform(-0.25, 0.25, size=CHAR_EMBED_SIZE),
    }
    with open("origin_data/char-embeddings.txt", "r", encoding="utf8") as f:
        for line in f:
            w, *values = line.strip().split()
            values = np.array(values, dtype='float32')
            char2vec[w] = values
    return char2vec


class SemEvalParser(HTMLParser):
    def __init__(self, char2vec):
        super(SemEvalParser, self).__init__()
        self.char2vec = char2vec
        self.unknown_chars = set()
        self.max_word_len = 0

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

    def extract_chars_feature(self):
        self.chars = []
        for i in range(SEQUENCE_LEN):
            if i < len(self.tokens):
                self.chars.append(self.char_embed(self.tokens[i]))
            else:
                self.chars.append(np.zeros(WORD_EMBED_SIZE))

    def feed(self, data):
        data = data.strip().split("\n")[0]
        data = data.strip().split("\t")[1][1:-1]

        self.data = []
        self.e1 = None
        self.e2 = None
        super(SemEvalParser, self).feed(data)

        self.tokens = word_tokenize(" ".join(self.data))
        self.tokens = [w[3:].replace("_", " ") if w == self.e1 or w == self.e2 else w for w in self.tokens]
        self.max_word_len = max(self.max_word_len, max(len(t) for t in self.tokens))

        self.extract_chars_feature()

    def char_embed(self, w):
        embed = []
        for i in range(WORD_LEN):
            if i < len(w):
                if w[i] in self.char2vec:
                    embed.append(self.char2vec[w[i]])
                else:
                    self.unknown_chars.add(w[i])
                    embed.append(self.char2vec["UNKNOWN"])
            else:
                embed.append(np.zeros(CHAR_EMBED_SIZE))
        return embed



def read_file(path, parser):
    chars = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            parser.feed(record)
            chars.append(parser.chars)

    return chars


def main():
    for folder in ["data", "data/train", "data/test"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print("read char embeddings")
    char2vec = read_char_embeddings()
    parser = SemEvalParser(char2vec)

    print("read train data")
    chars_train = read_file("origin_data/TRAIN_FILE.TXT", parser)
    np.save("data/train/chars.npy", chars_train)

    print("read test data")
    chars_test = read_file("origin_data/TEST_FILE_FULL.TXT", parser)
    np.save("data/test/chars.npy", chars_test)

    print("max_word_len: %d, unknown chars: %d" % (parser.max_word_len, len(parser.unknown_chars)))


if __name__ == "__main__":
    main()
