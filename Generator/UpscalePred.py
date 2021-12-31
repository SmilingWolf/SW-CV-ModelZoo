import cv2
import numpy as np

from tensorflow.keras.utils import Sequence


def smart_imread(img_path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), flag)
    if img is None:
        print("Error reading ", img_path)
    return img


def make_square(img):
    old_size = img.shape[:2]
    desired_size = max(old_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


class DataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(self, images_list, batch_size=16, dim=(48, 48), n_channels=3):
        "Initialization"
        self.dim = dim
        self.images_list = images_list
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.images_list) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X = self.__data_generation(indexes)

        return (X,)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        # Generate all the indexes
        self.indexes = np.arange(len(self.images_list))

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        # Generate data
        X = np.empty((len(indexes), self.dim[0], self.dim[1], self.n_channels))

        # Find list of IDs
        images_list_temp = [self.images_list[k] for k in indexes]

        for i, img_fullpath in enumerate(images_list_temp):
            img = smart_imread(img_fullpath)
            if img.dtype is np.dtype(np.uint16):
                img = (img / 257).astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                trans_mask = img[:, :, 3] == 0
                img[trans_mask] = [255, 255, 255, 255]
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img = make_square(img)
            if img.shape[0] > self.dim[0]:
                img = cv2.resize(
                    img, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_AREA
                )
            elif img.shape[0] < self.dim[0]:
                img = cv2.resize(
                    img, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_CUBIC
                )

            X[i] = img

        X = X.astype(np.float32) / 255

        return X
