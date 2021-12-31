import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from Utils import dbimutils


def representative_dataset_gen(images=None):
    dim = 320

    if images is None:
        images = open("2020_0000_0599/origlist.txt", "r").readlines()
        images = [r"D:\Images\danbooru2020\original\%s" % x.rstrip() for x in images]

    for image_path in images:
        img = dbimutils.smart_imread(image_path)
        img = dbimutils.smart_24bit(img)
        img = dbimutils.make_square(img, dim)
        img = dbimutils.smart_resize(img, dim)
        img = img.astype(np.float32) / 255
        yield img


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_images_list):
    global test_images

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file), num_threads=8)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    generator = representative_dataset_gen(test_images_list)
    for test_image in generator:
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            output_scale, output_zero_point = output_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        predictions.append(output * output_scale + output_zero_point)

    return np.array(predictions)


dim = 320
thresh = 0.3485
test_images = []

# images = open('2020_0000_0599/origlist.txt', 'r').readlines()
# images = ['D:\\Images\\danbooru2020\\original\\%s' % x.rstrip() for x in images]
images = ["82148729_p0.jpg"]

label_names = pd.read_csv("2020_0000_0599/selected_tags.csv")

probs = run_tflite_model("networks_tflite/NFNetL1V1-100-0.57141_u08.tflite", images)

label_names["probs"] = probs[0]
found_tags = label_names[label_names["probs"] > thresh][["tag_id", "name", "probs"]]

print(found_tags)
