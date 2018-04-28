# new_data = []
# count_other = 0
# with open("origin_data/data.cln", "r", encoding="utf8") as f:
#     lines = f.readlines()
# for line in lines:
#     if line[0] in ["0", "1", "2", "3"]:
#         new_data.append(line)
#     elif count_other < 400:
#         new_data.append(line)
#         count_other += 1
# with open("origin_data/new_data.cln", "w", encoding="utf8") as f:
#     f.writelines(new_data)

counter = {}
# with open("origin_data/data.cln", "r", encoding="utf8") as f:
#     content = f.read()
# content = content.replace("\ufeff", "")
# with open("origin_data/data.cln", "w", encoding="utf8") as f:
#     f.write(content)
with open("origin_data/data.cln", "r", encoding="utf8") as f:
    lines = f.readlines()
for line in lines:
    relation = line[0]
    if relation not in counter:
        counter[relation] = 0
    counter[relation] += 1
print(counter)
total = sum(counter.values())
print(total)