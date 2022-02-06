import cv2
import tensorflow as tf
from tqdm import tqdm

from Generator.ParseTFRecord import DataGenerator


def show(img):
    print(img.get_shape())
    cv2.imshow("test", img.numpy())
    cv2.waitKey(0)


total_labels = 2380

data_gen = DataGenerator(
    "record_shards/*",
    total_labels=total_labels,
    image_size=320,
    batch_size=32,
    noise_level=2,
    mixup_alpha=0.2,
    random_resize_method=True,
)

ds_iterator = iter(data_gen.genDS())
while 1:
    data = next(ds_iterator)
    show(data[0][0])

# for _ in tqdm(data_gen.genDS()):
#    pass
