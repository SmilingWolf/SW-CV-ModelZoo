import sqlite3

import pandas as pd
from tqdm import tqdm

all_files = open("danboorufiles.txt", "r").readlines()

df = pd.read_csv("purged.csv")
top_labels = set(df[df["category"] == 0]["tag_id"])

db = sqlite3.connect(r"F:\MLArchives\danbooru2021\danbooru2021.db")
db_cursor = db.cursor()

accepted = []
query = "SELECT tag_id FROM imageTags WHERE image_id = ?"
for img in tqdm(all_files):
    img_id = int(img.rsplit("/", 1)[1].rsplit(".", 1)[0])
    db_cursor.execute(query, (img_id,))
    tags = db_cursor.fetchall()
    tags = [tag_id[0] for tag_id in tags]
    top_tags = set(tags) & top_labels
    if len(top_tags) >= 10:
        accepted.append(img)

db.close()

with open("danboorufiles.txt", "w") as f:
    for line in accepted:
        f.write(line)
