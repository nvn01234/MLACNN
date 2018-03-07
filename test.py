import os
import time


def gen_answer_key(y_pred, meta):
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

    if not os.path.exists("log"):
        os.makedirs("log")
    with open("log/predict_keys.txt", "w", encoding="utf8") as f:
        for idx, y in zip(test_idx, y_pred):
            f.write("%s\t%s\n" % (idx, idx2relations[y]))

    timestamp = int(time.time())
    os.system("perl scorer.pl log/predict_keys.txt origin_data/test_keys.txt > log/result_%d.txt" % timestamp)
    os.remove("log/predict_keys.txt")
    with open("log/result_%d.txt" % timestamp, "a") as f:
        f.write(meta)
    with open("log/result_%d.txt" % timestamp, "r") as f:
        print(f.read())
    with open("log/results.txt", "a+") as f:
        f.write("\n\n==================%s=================\n\n%s" % (timestamp, meta))

