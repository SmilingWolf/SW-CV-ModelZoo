import numpy as np
import pandas as pd

threshold = 0.3485
tag_name = "virtual_youtuber"
path_prefix = r"D:\Images\danbooru2020\original"

files = [x.rstrip() for x in open("2020_0000_0599/origlist.txt").readlines()]
arr = np.load("2020_0000_0599/encoded_tags_test.npy", allow_pickle=True)
df = pd.read_csv("2020_0000_0599/selected_tags.csv")

index = np.where(df["name"] == tag_name)[0][0]

images_indexes = np.where(arr[:, index] > threshold)
image_paths = [files[x] for x in images_indexes[0]]

for partial_path in image_paths:
    print('cp "%s\\%s" test' % (path_prefix, partial_path.replace("/", "\\")))
