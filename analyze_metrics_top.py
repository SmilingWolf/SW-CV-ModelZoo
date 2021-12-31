import numpy as np
from matplotlib import pyplot as plt


def find_factor(img_tags, img_probs, start=0.3, end=0.9, points=6001):
    prec = []
    rec = []
    f1s = []
    f2s = []
    clip_points = []
    first_time = False
    yz = (img_tags > 0).astype(np.uint)
    for x in np.linspace(start, end, points):
        pos = (img_probs > x).astype(np.uint)
        pct = pos + 2 * yz

        TN = np.sum(pct == 0).astype(np.float32)
        FP = np.sum(pct == 1).astype(np.float32)
        FN = np.sum(pct == 2).astype(np.float32)
        TP = np.sum(pct == 3).astype(np.float32)

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        if precision >= recall and first_time == False:
            factor = round(x, 4)
            break
            print(x)
            first_time = True

        F1 = 2 * (precision * recall) / (precision + recall)
        F2 = 5 * (precision * recall) / ((4 * precision) + recall)

        clip_points.append(x)
        prec.append(precision)
        rec.append(recall)
        f1s.append(F1)
        f2s.append(F2)

    # plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    # plt.plot(clip_points, prec, label='precision')
    # plt.plot(clip_points, rec, label='recall')
    # plt.plot(clip_points, f1s, label='F1')
    # plt.plot(clip_points, f2s, label='F2')
    # plt.legend()
    # plt.show()
    return factor


img_probs = np.load("tags_probs_NFNetL1V1-100-0.57141.npy")
img_tags = np.load("2020_0000_0599/encoded_tags_test.npy")

img_probs = img_probs[:, 1092:]
img_tags = img_tags[:, 1092:]

"""
factor_end = 0.9
factor_start = 0.3
points = factor_end*100 - factor_start*100 + 1
factor = find_factor(img_tags, img_probs, factor_start, factor_end, int(points))
factor_start_step = factor * 100 - 1
factor_start = factor_start_step / 100
points = factor_end*1000 - factor_start*1000 + 1
factor = find_factor(img_tags, img_probs, factor_start, factor_end, int(points))
factor_start_step = factor * 1000 - 1
factor_start = factor_start_step / 1000
points = factor_end*10000 - factor_start*10000 + 1
factor = find_factor(img_tags, img_probs, factor_start, factor_end, int(points))
"""

factor_L1V1_100L = 0.3485
factor = factor_L1V1_100L
pos = (img_probs > factor).astype(np.uint)
yz = (img_tags > 0).astype(np.uint)
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

d = {
    "thres": factor,
    "F1": round(F1, 4),
    "F2": round(F2, 4),
    "MCC": round(MCC, 4),
    "A": round(accuracy, 4),
    "R": round(recall, 4),
    "P": round(precision, 4),
}
print(d)
