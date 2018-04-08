import os
import time


def gen_key(path, data, idx2relations):
    with open(path, "w", encoding="utf8") as f:
        for idx, y in enumerate(data):
            f.write("%s\t%s\n" % (idx, idx2relations[y]))


def evaluate(y_pred, y_true, fold):
    print("read relations file")
    idx2relations = {}
    with open("origin_data/relations.txt", "r", encoding="utf8") as f:
        for line in f:
            idx, r = line.strip().split()
            idx2relations[int(idx)] = r

    if not os.path.exists("log"):
        os.makedirs("log")
    gen_key("log/test_keys.txt", y_true, idx2relations)
    gen_key("log/predict_keys.txt", y_pred, idx2relations)

    os.system("perl scorer.pl log/predict_keys.txt log/test_keys.txt > log/result_%d.txt" % fold)
    os.remove("log/test_keys.txt")
    os.remove("log/predict_keys.txt")
    with open("log/result_%d.txt" % fold, "r") as f:
        f1_score = float(f.read().strip()[-10:-5])
    print("f1_score: %.2f" % f1_score)
    return f1_score

