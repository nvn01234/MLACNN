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
        for line in f:
            idx = line.strip().split()[0]
            test_idx.append(idx)

    lines = []
    for idx, y in zip(test_idx, y_pred):
        line = "%s\t%s" % (idx, idx2relations[y])
        lines.append(line)
    with open("log/predict_keys.txt", "w", encoding="utf8") as f:
        f.write("\n".join(lines))


def main():
    y_pred = np.load("log/y_pred.npy")
    gen_answer_key(y_pred)


if __name__ == "__main__":
    main()
