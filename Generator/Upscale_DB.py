import sqlite3

import cv2
import numpy as np
import tensorflow as tf


def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

    # return the rotated image
    return rotated


def crop(img, x, y, h, w):
    crop_img = img[y : y + h, x : x + w]
    return crop_img


class DataGenerator:
    def __init__(
        self, images_list, labels_list, noise_level=0, dim=(48, 48), n_channels=3
    ):
        self.dim = dim
        self.images_list = images_list
        self.labels_list = labels_list
        self.noise_level = noise_level
        self.n_channels = n_channels

    def getLabels(self, filename):
        db = sqlite3.connect(r"F:\MLArchives\danbooru2020\danbooru2020.db")
        db_cursor = db.cursor()

        img_id = int(
            filename.numpy().decode("utf-8").rsplit("/", 1)[1].rsplit(".", 1)[0]
        )

        query = "SELECT tag_id FROM imageTags WHERE image_id = ?"
        db_cursor.execute(query, (img_id,))
        tags = db_cursor.fetchall()
        db.close()

        tags = [tag_id[0] for tag_id in tags]
        encoded = np.isin(self.labels_list, tags).astype(np.float32)
        return encoded

    def getImage(self, filename):
        img_fullpath = r"F:\MLArchives\danbooru2020\512px\%s" % filename.numpy().decode(
            "utf-8"
        )
        img = cv2.imread(img_fullpath, cv2.IMREAD_COLOR)

        if self.noise_level >= 1:
            if np.random.choice([0, 1]):
                img = cv2.flip(img, 1)

            if np.random.choice([0, 1]) or self.noise_level >= 2:
                factor = (1.0 - 0.87) * np.random.random_sample() + 0.87
                origSize = img.shape[0]
                newSize = int(origSize * factor)
                if factor < 1.0:
                    x = np.random.randint(0, img.shape[1] - newSize)
                    y = np.random.randint(0, img.shape[0] - newSize)
                    img = crop(img, x, y, newSize, newSize)

            if np.random.choice([0, 1]) or self.noise_level >= 2:
                angle = np.random.randint(-45, 45)
                img = rotate(img, angle)

        img = cv2.resize(img, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_AREA)
        img = img * np.array(1 / 255.0).astype(np.float32)
        return img

    def wrap_func(self, filename):
        [
            image,
        ] = tf.py_function(self.getImage, [filename], [tf.float32])
        [
            image_labels,
        ] = tf.py_function(self.getLabels, [filename], [tf.float32])
        image.set_shape((self.dim[0], self.dim[1], self.n_channels))
        image_labels.set_shape(len(self.labels_list))
        return image, image_labels

    def genDS(self):
        return tf.data.Dataset.from_tensor_slices(self.images_list)
