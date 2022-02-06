import sqlite3

import numpy as np
import pandas as pd

all_files = open("danboorufiles.txt", "r").readlines()
top_tags = pd.read_csv("purged.csv")
top_counts = list(top_tags["count"])
top_ids = list(top_tags["tag_id"])

db = sqlite3.connect(r"F:\MLArchives\danbooru2020\danbooru2020.db")
db_cursor = db.cursor()

count = 0
totalTags = 0
tagCounts = dict.fromkeys(top_ids, 0)
for img in all_files:
    img_id = int(img.rsplit("/", 1)[1].rsplit(".", 1)[0])
    query = "SELECT tag_id FROM imageTags WHERE image_id = ?"
    db_cursor.execute(query, (img_id,))
    tags = db_cursor.fetchall()
    tags = [tag_id[0] for tag_id in tags]
    top_labels = np.intersect1d(top_ids, tags)
    totalTags += len(top_labels)
    count += 1
    for elem in top_labels:
        tagCounts[elem] += 1

db.close()

tagRatios = dict.fromkeys(top_ids, 0)
ratio = len(all_files) / 4226544
for top_id, top_count in zip(top_ids, top_counts):
    tagRatios[top_id] = int(top_count * ratio)

stuff = sorted(tagCounts, key=lambda x: tagCounts[x], reverse=True)
for elem in stuff:
    if tagCounts[elem] < tagRatios[elem] * 0.5 or tagCounts[elem] < 600:
        top_tags = top_tags[top_tags.tag_id != elem]
        print(elem, tagCounts[elem], tagRatios[elem])

top_tags.to_csv("purged.csv", index=False)

print(totalTags / count)
