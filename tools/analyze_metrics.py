import argparse

import numpy as np
import pandas as pd
import re

def filter_old_new(img_tags, img_probs, is_new=False):
    """
    is_new = True if model has been trained on danbooru2021
    is_new = False if model has been trained on danbooru2020
    """
    df2020 = pd.read_csv("2020_0000_0599/selected_tags.csv")
    df2021 = pd.read_csv("2021_0000_0899_5500/selected_tags.csv")

    names2020 = df2020["name"].tolist()
    names2021 = df2021["name"].tolist()

    index2020 = []
    index2021 = []
    for index, name in enumerate(names2021):
        if name in names2020:
            index2020.append(names2020.index(name))
            index2021.append(index)

    img_tags = img_tags[:, index2021]
    if is_new:
        img_probs = img_probs[:, index2021]
    else:
        img_probs = img_probs[:, index2020]

    return img_tags, img_probs


def calc_metrics(img_tags, img_probs, thresh):
    yz = (img_tags > 0).astype(np.uint8)
    pos = (img_probs > thresh).astype(np.uint8)
    pct = pos + 2 * yz

    TN = np.sum(pct == 0).astype(np.float32)
    FP = np.sum(pct == 1).astype(np.float32)
    FN = np.sum(pct == 2).astype(np.float32)
    TP = np.sum(pct == 3).astype(np.float32)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return precision, recall


parser = argparse.ArgumentParser(description="Analyze output probabilities dumps")
parser.add_argument(
    "-d",
    "--dump",
    default="data/tags_probs_NFNetL2V1_01_29_2022_08h08m56s.npy",
    help="Numpy dump to use",
)

parser.add_argument(
    "-b",
    "--bottom-index",
    type=int,
    default=0,
    help="Slice files along axis=1 starting from this index",
)

parser.add_argument(
    "-c",
    "--category",
    type=int,
    default=-1,
    help="Only analyze tags of this category (-1 = all)",
)

thresh_group = parser.add_mutually_exclusive_group()
thresh_group.add_argument(
    "-a",
    "--analyze",
    action="store_true",
    help="Iteratively look for the threshold where P ≈ R",
)
thresh_group.set_defaults(analyze=False)

thresh_group.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.4,
    help="Use this threshold to calculate the metrics",
)
args = parser.parse_args()

img_probs = np.load(args.dump)
img_tags = np.load("2021_0000_0899_5500/encoded_tags_test.npy")

if args.category > -1:
    df = pd.read_csv("2021_0000_0899_5500/selected_tags.csv")
    indexes = np.where(df["category"] == args.category)[0]
    img_probs = img_probs[:, indexes]
    img_tags  = img_tags[:, indexes]

# img_tags, img_probs = filter_old_new(img_tags, img_probs, True)

if args.bottom_index > 0:
    img_probs = img_probs[:, args.bottom_index :]
    img_tags = img_tags[:, args.bottom_index :]

if args.analyze:
    threshold_min = 0.1
    threshold_max = 0.95

    recall = 0.0
    precision = 1.0
    while not np.isclose(recall, precision):
        threshold = (threshold_max + threshold_min) / 2
        precision, recall = calc_metrics(img_tags, img_probs, threshold)
        if precision > recall:
            threshold_max = threshold
        else:
            threshold_min = threshold

    threshold = round(threshold, 4)
else:
    threshold = args.threshold

pos = (img_probs > threshold).astype(np.uint8)
yz = (img_tags > 0).astype(np.uint8)
pct = pos + 2 * yz

TN = np.sum(pct == 0).astype(np.float32)
FP = np.sum(pct == 1).astype(np.float32)
FN = np.sum(pct == 2).astype(np.float32)
TP = np.sum(pct == 3).astype(np.float32)

recall = TP / (TP + FN)
precision = TP / (TP + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)

F1 = 2 * (precision * recall) / (precision + recall)
F2 = 5 * (precision * recall) / ((4 * precision) + recall)

MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

model_name = re.sub(".*tags_probs_", "", args.dump)
model_name = model_name.replace(".npy", "")
d = {
    "thres": threshold,
    "F1": round(F1, 4),
    "F2": round(F2, 4),
    "MCC": round(MCC, 4),
    "A": round(accuracy, 4),
    "R": round(recall, 4),
    "P": round(precision, 4),
}
print(f"{model_name}: {str(d)}")
