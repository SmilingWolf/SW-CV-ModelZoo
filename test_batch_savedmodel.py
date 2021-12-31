import os

import cv2
import numpy as np

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
images_basepath = r"F:\MLArchives\danbooru2020\512px"

imagesList = open("2020_0000_0599/testlist.txt").readlines()
imagesList = [x.rstrip() for x in imagesList]
imagesList = ["%s/%s" % (images_basepath, x) for x in imagesList]

model = NFNetV1(
    in_shape=(dim, dim, 3), out_classes=2380, definition_name="L1", use_eca=False
)

for modelNum in [100]:
    model.load_weights("trial/NFNetL1V1-rmc-%02d/variables/variables" % modelNum)
    model.trainable = False

    generator = DataGenerator(imagesList, batch_size=32, dim=(dim, dim))

    probs = model.predict(generator, verbose=1, use_multiprocessing=False, workers=7)

    np.save("tags_probs_%dL.npy" % modelNum, probs)
