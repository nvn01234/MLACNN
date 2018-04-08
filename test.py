import os
import time


def gen_key(path, data, idx2relations):
    with open(path, "w", encoding="utf8") as f:
        for idx, y in enumerate(data):
            f.write("%s\t%s\n" % (idx, idx2relations[y]))


def evaluate(y_pred, y_true, result_path=None):
    idx2relations = {}
    with open("origin_data/relations.txt", "r", encoding="utf8") as f:
        for line in f:
            idx, r = line.strip().split()
            idx2relations[int(idx)] = r

    log_dir = "log"
    test_keys_path = os.path.join(log_dir, "test_keys.txt")
    predict_keys_path = os.path.join(log_dir, "predict_keys.txt")

    os.makedirs(log_dir, exist_ok=True)
    gen_key(test_keys_path, y_true, idx2relations)
    gen_key(predict_keys_path, y_pred, idx2relations)

    if result_path is None:
        result_path = os.path.join(log_dir, "tmp_result.txt")
        keep_result = False
    else:
        keep_result = True

    os.system("perl scorer.pl %s %s > %s" % (predict_keys_path, test_keys_path, result_path))
    with open(result_path, "r") as f:
        f1_score = float(f.read().strip()[-10:-5])

    os.remove(test_keys_path)
    os.remove(predict_keys_path)
    if not keep_result:
        os.remove(result_path)

    return f1_score

