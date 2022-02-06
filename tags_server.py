import base64
import json
import os

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from Utils import dbimutils

use_GPU = False
if use_GPU == False:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf

from Models.NFNet import NFNetV1


class JitModel:
    def __init__(self, model):
        self.model = model

    @tf.function
    def predict(self, x):
        return self.model(x, training=False)


app = Flask(__name__)


@app.route("/api/gettags/", methods=["POST"])
def gettags():
    """
    Function run at each API call
        No need to re-load the model
    """

    request_json = request.get_json()

    thresh = 0.3228 if "thresh" not in request_json else float(request_json["thresh"])
    img = request_json["image"]
    txt_img = base64.b64decode(img)
    np_img = np.frombuffer(txt_img, np.uint8)
    img = cv2.imdecode(np_img, flags=cv2.IMREAD_UNCHANGED)
    img = dbimutils.smart_24bit(img)
    img = dbimutils.make_square(img, dim)
    img = dbimutils.smart_resize(img, dim)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    probs = model.predict(img).numpy()

    label_names["probs"] = probs[0]
    found_tags = label_names[label_names["probs"] > thresh][["name", "category"]]

    labels_list = []
    for index, pair in found_tags.iterrows():
        if pair["category"] == 0:
            labels_list.append(pair["name"])
        elif pair["category"] == 3:
            labels_list.append("series:%s" % pair["name"])
        elif pair["category"] == 4:
            labels_list.append("character:%s" % pair["name"])

    return jsonify(labels_list)


if __name__ == "__main__":
    # Model is loaded when the API is launched
    dim = 320
    model = NFNetV1(
        in_shape=(dim, dim, 3),
        out_classes=2380,
        definition_name="L1",
        cnn_attention="eca",
        compensate_avgpool_var=True,
        activation="silu",
    )
    model.load_weights(r"networks\NFNetL1V1_01_29_2022_08h20m44s\variables\variables")
    model = JitModel(model)

    label_names = pd.read_csv("2021_0000_0899/selected_tags.csv")

    app.run(debug=False)
