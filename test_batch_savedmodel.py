import os

import numpy as np

use_GPU = True
if use_GPU == False:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

from Generator.UpscalePred import DataGenerator

dim = 320
images_basepath = r"F:\MLArchives\danbooru2021\512px"

imagesList = open("2021_0000_0899/testlist.txt").readlines()
imagesList = [x.rstrip() for x in imagesList]
imagesList = ["%s/%s" % (images_basepath, x) for x in imagesList]

for model_name in ["NFNetL1V1_01_29_2022_08h20m44s"]:
    model = tf.keras.models.load_model("checkpoints/%s" % model_name)
    model.trainable = False

    generator = DataGenerator(imagesList, batch_size=32, dim=(dim, dim))

    probs = model.predict(generator, verbose=1, use_multiprocessing=False, workers=7)

    np.save("tags_probs_%s.npy" % model_name, probs)

    del model
    tf.keras.backend.clear_session()
