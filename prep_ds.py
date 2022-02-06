import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from Generator.PrepTFRecord import DataGenerator


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=tf.reshape(value, (-1,)))
    )


def serialize_example(image_id, image_bytes, label_indexes):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "image_id": _int64_feature(image_id),
        "image_bytes": _bytes_feature(image_bytes),
        "label_indexes": _int64_feature(label_indexes),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def generator():
    for features in training_dataset:
        yield serialize_example(*features)


# -----
num_shards = 91
records_per_shard = 1024
train_list = open("2021_0000_0899/testlist.txt", "r").readlines()[
    : records_per_shard * num_shards
]
train_list = [x.rstrip() for x in train_list]
train_list = ["/mnt/data/danbooru2021/512px/%s" % x for x in train_list]

labels_list = pd.read_csv("2021_0000_0899/selected_tags.csv")["tag_id"].tolist()

training_generator = DataGenerator(
    "/mnt/data/danbooru2021.db",
    train_list,
    labels_list,
)

training_dataset = training_generator.genDS()
training_dataset = training_dataset.map(training_generator.wrap_func)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=()
)

serial_gen = generator()
for shard_index in tqdm(range(num_shards)):
    with tf.io.TFRecordWriter(
        "gs://danbooru2021_sfw/record_shards_val/danbooru2021_top2380_shard%04d.tfrecord"
        % shard_index
    ) as file_writer:
        shard_records = 0
        while shard_records < records_per_shard:
            record_bytes = next(serial_gen)
            file_writer.write(record_bytes)
            shard_records += 1
# -----
num_shards = 2749
records_per_shard = 1024
train_list = open("2021_0000_0899/trainlist.txt", "r").readlines()[
    : records_per_shard * num_shards
]
train_list = [x.rstrip() for x in train_list]
train_list = ["/mnt/data/danbooru2021/512px/%s" % x for x in train_list]

labels_list = pd.read_csv("2021_0000_0899/selected_tags.csv")["tag_id"].tolist()

training_generator = DataGenerator(
    "/mnt/data/danbooru2021.db",
    train_list,
    labels_list,
)

training_dataset = training_generator.genDS()
training_dataset = training_dataset.map(training_generator.wrap_func)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=()
)

serial_gen = generator()
for shard_index in tqdm(range(num_shards)):
    with tf.io.TFRecordWriter(
        "gs://danbooru2021_sfw/record_shards_train/danbooru2021_top2380_shard%04d.tfrecord"
        % shard_index
    ) as file_writer:
        shard_records = 0
        while shard_records < records_per_shard:
            record_bytes = next(serial_gen)
            file_writer.write(record_bytes)
            shard_records += 1
