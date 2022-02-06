import os
import sys

import numpy as np
import onnxruntime as rt
import pandas as pd

from Utils import dbimutils

pd.set_option("display.max_rows", 1000)

dim = 320
thresh = 0.3228
model = rt.InferenceSession("networks/NFNetL1V1_01_29_2022_08h20m44s.onnx")
label_names = pd.read_csv("2021_0000_0899/selected_tags.csv")

target_img = "82148729_p0.jpg" if len(sys.argv) < 2 else sys.argv[1]

img = dbimutils.smart_imread(target_img)
img = dbimutils.smart_24bit(img)
img = dbimutils.make_square(img, dim)
img = dbimutils.smart_resize(img, dim)
img = img.astype(np.float32)
img = np.expand_dims(img, 0)

input_name = model.get_inputs()[0].name
label_name = model.get_outputs()[0].name
probs = model.run([label_name], {input_name: img})[0]

label_names["probs"] = probs[0]
found_tags = label_names[label_names["probs"] > thresh][["tag_id", "name", "probs"]]

print(found_tags)
