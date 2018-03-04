import numpy as np
import os


def label2idx(label):
    labels_mapping = {'Other': 0,
                      'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                      'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                      'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                      'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                      'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                      'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                      'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                      'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                      'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
    if label in labels_mapping:
        label = labels_mapping[label]
    else:
        print(label)
        label = labels_mapping["Other"]
    return label


def read_file(path):
    labels = []
    with open(path, "r", encoding="utf8") as f:
        records = f.read().strip().split("\n\n")
        for record in records:
            label = record.strip().split("\n")[1]
            labels.append(label2idx(label))

    return labels


def main():
    for folder in ["data", "data/train", "data/test"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print("read train data")
    labels_train = read_file("origin_data/TRAIN_FILE.TXT")
    np.save("data/train/labels.npy", labels_train)

    print("read test data")
    labels_test = read_file("origin_data/TEST_FILE_FULL.TXT")
    np.save("data/test/labels.npy", labels_test)


if __name__ == "__main__":
    main()
