import os


def gen_answer_key(y_pred, name):
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

    with open("log/predict_keys_%s.txt" % name, "w", encoding="utf8") as f:
        for idx, y in zip(test_idx, y_pred):
            f.write("%s\t%s\n" % (idx, idx2relations[y]))



def main():
    name = "base"
    os.system("perl scorer.pl log/predict_keys_{0}.txt origin_data/test_keys.txt > log/result_scores_{0}.txt".format(name))
    with open("log/result_scores_%s.txt" % name, "r") as f:
        result = f.read()
    print(result)


if __name__ == "__main__":
    main()
