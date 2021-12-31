import pathlib

import cv2
import numpy as np

from Utils import dbimutils


def representative_dataset_gen():
    dim = 320
    images = open("2020_0000_0599/origlist.txt", "r").readlines()
    images = [r"D:\Images\danbooru2020\original\%s" % x.rstrip() for x in images]
    for i in range(8375):
        target_img = images[i]
        img = dbimutils.smart_imread(target_img)
        img = dbimutils.smart_24bit(img)
        img = dbimutils.make_square(img, dim)
        img = dbimutils.smart_resize(img, dim)
        img = img.astype(np.float64)
        yield img


means = np.zeros((3,))
stds = np.zeros((3,))
img_gen = representative_dataset_gen()
for img in img_gen:
    means += img.mean(axis=(0, 1))
    stds += img.std(axis=(0, 1))
print(np.around((means / 8375) / 255, 3), np.around((stds / 8375) / 255, 3))

# output:
# mean = np.array([0.747, 0.752, 0.792])
# std = np.array([0.262, 0.265, 0.249])
