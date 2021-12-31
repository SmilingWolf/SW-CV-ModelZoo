import numpy as np
import pandas as pd

tags = pd.read_csv("2020_0000_0599/selected_tags.csv")
label = np.load("2020_0000_0599/encoded_tags_test.npy")

factor = 0.3485
eps = np.finfo(np.float32).eps
preds = np.load("tags_probs_NFNetL1V1-100-0.57141.npy")
preds = (preds > factor).astype(np.uint8)

actual_freq = np.sum(label, axis=0) / label.shape[0]
predicted_freq = np.sum(preds, axis=0) / label.shape[0]

pct = preds + 2 * label
TN = np.sum(pct == 0, axis=0).astype(np.float32)
FP = np.sum(pct == 1, axis=0).astype(np.float32)
FN = np.sum(pct == 2, axis=0).astype(np.float32)
TP = np.sum(pct == 3, axis=0).astype(np.float32)

recall = TP / (TP + FN)
precision = TP / ((TP + FP) + eps)

F1 = 2 * (precision * recall) / ((precision + recall) + eps)

df = pd.DataFrame()
df["Tag"] = tags["name"]
df["Category"] = tags["category"]
df["Actual_posts"] = np.sum(label, axis=0)
df["Predicted_posts"] = np.sum(preds, axis=0)
df["Correct_predictions"] = TP.astype(np.uint)
df["Actual_frequency"] = actual_freq
df["Predicted_frequency"] = predicted_freq
df["Precision"] = precision
df["Recall"] = recall
df["F_score"] = F1

df = df.sort_values("Actual_posts", ascending=False)
df.to_csv("report.csv", index=False, float_format="%.02f")
df.to_html("report.html", index=False, float_format="%.02f")
