import os

import cv2
import numpy as np
import pandas as pd

use_GPU = True
if use_GPU == False:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

from Generator.UpscalePred import DataGenerator
from Models.NFNet import NFNetV1

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

dim = 320
thresh = 0.3485
images_folder = r"C:\images"

label_names = pd.read_csv("2020_0000_0599/selected_tags.csv")

model = NFNetV1(
    in_shape=(dim, dim, 3), out_classes=2380, definition_name="L1", use_eca=False
)
model.load_weights("networks/NFNetL1V1-100-0.57141/variables/variables")
model.trainable = False

images_list = []
for r, d, f in os.walk(images_folder):
    for file in f:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            images_list.append(os.path.join(r, file))

generator = DataGenerator(images_list, batch_size=4, dim=(dim, dim))

probs = model.predict(generator, verbose=1, use_multiprocessing=False, workers=7)

# Surely there must be a better way - and don't call me Shirley
indexes = [np.where(probs[x, :] > thresh)[0] for x in range(probs.shape[0])]

for image, index_list in zip(images_list, indexes):
    labels_list = []
    extracted = label_names.iloc[index_list][["name", "category"]]
    for index, pair in extracted.iterrows():
        if pair["category"] == 0:
            labels_list.append(pair["name"])
        # elif pair['category'] == 3:
        #    labels_list.append('series:%s' % pair['name'])
        # elif pair['category'] == 4:
        #    labels_list.append('character:%s' % pair['name'])
    labels_list = "\n".join(labels_list)
    with open("%s.txt" % image, "w") as f:
        f.writelines(labels_list)
