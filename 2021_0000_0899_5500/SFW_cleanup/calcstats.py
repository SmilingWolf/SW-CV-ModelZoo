import sqlite3

import pandas as pd
from tqdm import tqdm

all_files = open("danboorufiles.txt", "r").readlines()
top_tags = pd.read_csv("purged.csv")
top_counts = list(top_tags["count"])
top_ids = list(top_tags["tag_id"])
top_ids_set = set(top_ids)

db = sqlite3.connect(r"F:\MLArchives\danbooru2021\danbooru2021.db")
db_cursor = db.cursor()

count = 0
total_tags = 0
tag_counts = dict.fromkeys(top_ids, 0)
query = "SELECT tag_id FROM imageTags WHERE image_id = ?"
for img in tqdm(all_files):
    img_id = int(img.rsplit("/", 1)[1].rsplit(".", 1)[0])
    db_cursor.execute(query, (img_id,))
    tags = db_cursor.fetchall()
    tags = [tag_id[0] for tag_id in tags]
    top_labels = set(tags) & top_ids_set
    total_tags += len(top_labels)
    count += 1
    for elem in top_labels:
        tag_counts[elem] += 1

db.close()

tag_ratios = dict.fromkeys(top_ids, 0)
ratio = len(all_files) / 4863782
for top_id, top_count in zip(top_ids, top_counts):
    tag_ratios[top_id] = int(top_count * ratio)

stuff = sorted(tag_counts, key=lambda x: tag_counts[x], reverse=True)
for elem in stuff:
    if tag_counts[elem] < tag_ratios[elem] * 0.5 or tag_counts[elem] < 600:
        top_tags = top_tags[top_tags.tag_id != elem]
        # print(elem, tag_counts[elem], tag_ratios[elem])

top_tags.to_csv("purged.csv", index=False)

print(total_tags / count)
