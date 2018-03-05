import numpy as np


def gen_answer_key(y_pred):
    print("read relations file")
    idx2relations = {}
    with open("origin_data/relations.txt", "r", encoding="utf8") as f:
        for line in f:
            idx, r = line.strip().split()
            idx2relations[int(idx)] = r

    print("read test keys")
    test_idx = []
    with open("origin_data/test_keys.txt", "r", encoding="utf8") as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        idx = line.strip().split()[0]
        test_idx.append(idx)

    with open("log/predict_keys_2.txt", "w", encoding="utf8") as f:
        for idx, y in zip(test_idx, y_pred):
            f.write("%s\t%s\n" % (idx, idx2relations[y]))


def main():
    y_pred = np.load("log/y_pred.npy")
    gen_answer_key(y_pred)


if __name__ == "__main__":
    main()
