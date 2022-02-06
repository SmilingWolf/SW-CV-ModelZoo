import sqlite3

import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, db_path, images_list, labels_list):
        self.db_path = db_path
        self.images_list = images_list
        self.labels_list = labels_list

    def getLabels(self, filename):
        db = sqlite3.connect(self.db_path)
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
        return img_id, encoded

    def getImage(self, filename):
        img_bytes = tf.io.read_file(filename)
        return img_bytes

    def wrap_func(self, filename):
        image_bytes = self.getImage(filename)
        [image_id, image_labels_1h] = tf.py_function(
            self.getLabels, [filename], [tf.int64, tf.float32]
        )
        image_id.set_shape(())
        image_labels_1h.set_shape(len(self.labels_list))
        image_labels = tf.where(image_labels_1h)
        return image_id, image_bytes, image_labels

    def genDS(self):
        return tf.data.Dataset.from_tensor_slices(self.images_list)
