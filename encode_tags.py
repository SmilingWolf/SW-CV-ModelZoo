import numpy as np
import pandas as pd
import sqlite3

from tqdm import tqdm

df = pd.read_csv("2020_0000_0599/selected_tags.csv")
labels = df["tag_id"].tolist()

db = sqlite3.connect(r"F:\MLArchives\danbooru2020\danbooru2020.db")
db_cursor = db.cursor()

images_list = open("2020_0000_0599/testlist.txt").readlines()
img_ids = [int(image.rsplit("/", 1)[1].rsplit(".", 1)[0]) for image in images_list]

query = "SELECT tag_id FROM imageTags WHERE image_id = ?"
img_tags = np.empty((len(img_ids), len(labels)), dtype=np.uint8)
for index, img_id in enumerate(tqdm(img_ids)):
    db_cursor.execute(query, (img_id,))
    tags = db_cursor.fetchall()
    tags = [tag_id[0] for tag_id in tags]
    if len(tags) == 0:
        print("%s: found 0 tags" % image)
        continue
    encoded = np.isin(labels, tags).astype(np.uint8)
    img_tags[index] = encoded

db.close()
np.save("2020_0000_0599/encoded_tags_test.npy", img_tags)
