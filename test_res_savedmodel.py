import os
import sys

import numpy as np
import pandas as pd

from Utils import dbimutils

pd.set_option("display.max_rows", 1000)

use_GPU = True
if use_GPU == False:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from tensorflow.keras.models import load_model

from Models.NFNet import NFNetV1

dim = 320
thresh = 0.3228
# model = NFNetV1(
#     in_shape=(dim, dim, 3),
#     out_classes=2380,
#     definition_name="L1",
#     cnn_attention="eca",
#     compensate_avgpool_var=True,
#     activation="silu",
# )
# model.load_weights("checkpoints/NFNetL1V1_01_29_2022_08h20m44s/variables/variables")
model = load_model("networks/NFNetL1V1_01_29_2022_08h20m44s")
label_names = pd.read_csv("2021_0000_0899/selected_tags.csv")

target_img = "82148729_p0.jpg" if len(sys.argv) < 2 else sys.argv[1]

img = dbimutils.smart_imread(target_img)
img = dbimutils.smart_24bit(img)
img = dbimutils.make_square(img, dim)
img = dbimutils.smart_resize(img, dim)
img = img.astype(np.float32)
img = np.expand_dims(img, 0)

probs = model.predict(img)

label_names["probs"] = probs[0]
found_tags = label_names[label_names["probs"] > thresh][["tag_id", "name", "probs"]]

print(found_tags)
