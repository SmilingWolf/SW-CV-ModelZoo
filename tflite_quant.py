import pathlib

import cv2
import numpy as np
import tensorflow as tf

from Utils import dbimutils


def representative_dataset_gen():
    dim = 320
    images = open("2020_0000_0599/origlist.txt", "r").readlines()
    images = [r"D:\Images\danbooru2020\original\%s" % x.rstrip() for x in images]

    rng = np.random.default_rng(1249)
    rng.shuffle(images)
    for i in range(1000):
        target_img = images[i]
        img = dbimutils.smart_imread(target_img)
        img = dbimutils.smart_24bit(img)
        img = dbimutils.make_square(img, dim)
        img = dbimutils.smart_resize(img, dim)
        img = img.astype(np.float32) / 255
        img = np.expand_dims(img, 0)
        yield [img.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model("checkpoints/openvino-tflite")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("networks_tflite")
tflite_models_dir.mkdir(parents=True, exist_ok=True)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir / "NFResNet50V1-50-0.58250_u08.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
