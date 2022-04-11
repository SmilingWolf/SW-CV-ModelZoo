import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


class DataGenerator:
    def __init__(
        self,
        records_path,
        total_labels=2380,
        image_size=320,
        batch_size=32,
        noise_level=0,
        mixup_alpha=0.2,
        cutout_max_pct=0.25,
        random_resize_method=True,
    ):
        """
        Noise level 1: augmentations I will never train without
                       unless I'm dealing with extremely small networks
                       (Random rotation, random cropping and random flipping)

        Noise level 2: more advanced stuff (MixUp)
        """

        self.records_path = records_path
        self.total_labels = total_labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.mixup_alpha = mixup_alpha
        self.random_resize_method = random_resize_method

        self.cutout_max_pct = cutout_max_pct
        self.cutout_replace = 127

    def parse_single_record(self, example_proto):
        feature_description = {
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "image_bytes": tf.io.FixedLenFeature([], tf.string),
            "label_indexes": tf.io.VarLenFeature(tf.int64),
        }

        # Parse the input 'tf.train.Example' proto using the dictionary above.
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        image_tensor = tf.io.decode_jpeg(parsed_example["image_bytes"], channels=3)

        # RGB -> BGR (legacy reasons)
        image_tensor = tf.gather(image_tensor, axis=2, indices=[2, 1, 0])

        # Nel TFRecord mettiamo solo gli indici per questioni di spazio
        # Emula MultiLabelBinarizer a partire dagli indici per ottenere un tensor di soli 0 e 1
        label_indexes = tf.sparse.to_dense(
            parsed_example["label_indexes"], default_value=0
        )
        one_hots = tf.one_hot(label_indexes, self.total_labels)
        labels = tf.reduce_max(one_hots, axis=0)
        labels = tf.cast(labels, tf.float32)

        return image_tensor, labels

    def random_flip(self, image, labels):
        image = tf.image.random_flip_left_right(image)
        return image, labels

    def random_crop(self, image, labels):
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]

        factor = tf.random.uniform(shape=[], minval=0.87, maxval=0.998)

        # Assuming this is a standard 512x512 Danbooru20xx SFW image
        new_height = new_width = tf.cast(tf.cast(height, tf.float32) * factor, tf.int32)

        offset_height = tf.random.uniform(
            shape=[], minval=0, maxval=(height - new_height), dtype=tf.int32
        )
        offset_width = tf.random.uniform(
            shape=[], minval=0, maxval=(width - new_width), dtype=tf.int32
        )
        image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, new_height, new_width
        )
        return image, labels

    def random_rotate(self, images, labels):
        angles = (
            tf.random.uniform(shape=[self.batch_size], minval=-45, maxval=45)
            * np.pi
            / 180.0
        )
        images = tfa.image.rotate(
            images, angles, interpolation="bilinear", fill_value=255.0
        )
        return images, labels

    def resize(self, image, labels):
        if self.random_resize_method:
            # During training mix algos up to make the model a bit more more resilient
            # to the different image resizing implementations out there (TF, OpenCV, PIL, ...)
            method_index = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
            if method_index == 0:
                image = tf.image.resize(
                    images=image,
                    size=(self.image_size, self.image_size),
                    method="area",
                    antialias=True,
                )
            elif method_index == 1:
                image = tf.image.resize(
                    images=image,
                    size=(self.image_size, self.image_size),
                    method="bilinear",
                    antialias=True,
                )
            else:
                image = tf.image.resize(
                    images=image,
                    size=(self.image_size, self.image_size),
                    method="bicubic",
                    antialias=True,
                )
        else:
            image = tf.image.resize(
                images=image,
                size=(self.image_size, self.image_size),
                method="area",
                antialias=True,
            )
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
        return image, labels

    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    def cutout(self, image, labels):
        """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
        This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
        a random location within `img`. The pixel values filled in will be of the
        value `replace`. The located where the mask will be applied is randomly
        chosen uniformly over the whole image.
        Args:
          image: An image Tensor of type uint8.
          pad_size: Specifies how big the zero mask that will be generated is that
            is applied to the image. The mask will be of size
            (2*pad_size x 2*pad_size).
          replace: What pixel value to fill in the image in the area that has
            the cutout mask applied to it.
        Returns:
          An image Tensor that is of type uint8.
        """
        max_pct = self.cutout_max_pct
        replace = self.cutout_replace

        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        pad_pct = tf.random.uniform((), minval=0, maxval=max_pct)
        img_area = image_height * image_width
        pad_area = tf.cast(img_area, dtype=tf.float32) * pad_pct
        pad_size = tf.cast(tf.math.sqrt(pad_area) / 2, dtype=tf.int32)

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = tf.random.uniform(
            shape=[], minval=0, maxval=image_height, dtype=tf.int32
        )

        cutout_center_width = tf.random.uniform(
            shape=[], minval=0, maxval=image_width, dtype=tf.int32
        )

        lower_pad = tf.maximum(0, cutout_center_height - pad_size)
        upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = tf.maximum(0, cutout_center_width - pad_size)
        right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [
            image_height - (lower_pad + upper_pad),
            image_width - (left_pad + right_pad),
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(
            tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
        )
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 3])
        image = tf.where(
            tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image
        )
        return image, labels

    def mixup_single(self, images, labels):
        alpha = self.mixup_alpha

        # Unpack one dataset, generate a second by reversing the input one on the batch axis
        images_one, labels_one = tf.cast(images, tf.float32), tf.cast(
            labels, tf.float32
        )
        images_two = tf.reverse(images_one, axis=[0])
        labels_two = tf.reverse(labels_one, axis=[0])
        batch_size = tf.shape(images_one)[0]

        # Sample lambda and reshape it to do the mixup
        l = sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)

        images = tf.cast(tf.clip_by_value(images, 0, 255), tf.uint8)
        return images, labels

    def genDS(self):
        files = tf.data.Dataset.list_files(self.records_path)

        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x).map(
                self.parse_single_record, num_parallel_calls=1
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10 * self.batch_size)

        if self.noise_level >= 1:
            dataset = dataset.map(self.random_flip, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(self.random_crop, num_parallel_calls=tf.data.AUTOTUNE)

        # Resize before batching. Especially important if random_crop is enabled
        dataset = dataset.map(self.resize, num_parallel_calls=tf.data.AUTOTUNE)

        if self.noise_level >= 2 and self.cutout_max_pct > 0.0:
            dataset = dataset.map(self.cutout, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Rotation is very slow on CPU. Rotating a batch of resized images is much faster
        if self.noise_level >= 1:
            dataset = dataset.map(
                self.random_rotate, num_parallel_calls=tf.data.AUTOTUNE
            )

        if self.noise_level >= 2 and self.mixup_alpha > 0.0:
            dataset = dataset.map(
                self.mixup_single, num_parallel_calls=tf.data.AUTOTUNE
            )

        return dataset
