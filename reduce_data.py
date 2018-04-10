new_data = []
count_other = 0
with open("origin_data/data.cln", "r", encoding="utf8") as f:
    lines = f.readlines()
for line in lines:
    if line[0] in ["0", "1", "2", "3"]:
        new_data.append(line)
    elif count_other < 400:
        new_data.append(line)
        count_other += 1
with open("origin_data/new_data.cln", "w", encoding="utf8") as f:
    f.writelines(new_data)