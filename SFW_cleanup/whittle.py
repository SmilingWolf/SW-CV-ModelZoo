import sqlite3

import numpy as np
import pandas as pd

all_files = open("danboorufiles.txt", "r").readlines()
top_labels = list(pd.read_csv("purged.csv")["tag_id"])

db = sqlite3.connect(r"F:\MLArchives\danbooru2020\danbooru2020.db")
db_cursor = db.cursor()

accepted = []
for img in all_files:
    img_id = int(img.rsplit("/", 1)[1].replace(".jpg", ""))
    query = "SELECT tag_id FROM imageTags WHERE image_id = ?"
    db_cursor.execute(query, (img_id,))
    tags = db_cursor.fetchall()
    tags = [tag_id[0] for tag_id in tags]
    top_tags = np.intersect1d(top_labels, tags)
    if len(top_tags) >= 15:
        accepted.append(img)

db.close()

with open("danboorufiles.txt", "w") as f:
    for line in accepted:
        f.write(line)
