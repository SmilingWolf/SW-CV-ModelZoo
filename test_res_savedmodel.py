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
thresh = 0.3485
# model = NFNetV1(in_shape=(dim, dim, 3), out_classes=2380, definition_name="L1", use_eca=False)
# model.load_weights("networks/NFNetL1V1-100-0.57141/variables/variables")
model = load_model("networks/NFNetL1V1-100-0.57141")
label_names = pd.read_csv("2020_0000_0599/selected_tags.csv")

target_img = "82148729_p0.jpg" if len(sys.argv) < 2 else sys.argv[1]

img = dbimutils.smart_imread(target_img)
img = dbimutils.smart_24bit(img)
img = dbimutils.make_square(img, dim)
img = dbimutils.smart_resize(img, dim)
img = img.astype(np.float32) / 255
img = np.expand_dims(img, 0)

probs = model.predict(img)

label_names["probs"] = probs[0]
found_tags = label_names[label_names["probs"] > thresh][["tag_id", "name", "probs"]]

print(found_tags)
