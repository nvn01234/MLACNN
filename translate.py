from googletrans import Translator
import re
from tqdm import tqdm


def read_file(path):
    print("read %s" % path)
    translator = Translator()
    with open(path, "r", encoding="utf8") as f:
        content = f.read().strip()
    records = content.split("\n\n")
    count_success = 0
    count_fail = 0
    for record in tqdm(records):
        text, relation, _ = record.strip().split("\n")
        text = text.split("\t")[1][1:-2]

        entities = re.findall("<e[12]>([^<]+)</e[12]>", text)
        vi_entities = [translator.translate(e, dest="vi", src="en").text for e in entities]

        text = re.sub("</?e[12]>", "", text)
        vi_text = translator.translate(text, dest="vi", src="en").text
        success = all(e.lower() in vi_text.lower() for e in vi_entities)
        count_success += int(success)
        count_fail += int(not success)
    print("success: %d, fail: %d" % (count_success, count_fail))


if __name__ == "__main__":
    read_file("origin_data/TRAIN_FILE.TXT")
    read_file("origin_data/TEST_FILE_FULL.TXT")