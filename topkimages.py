import numpy as np
import pandas as pd

lines = open("2020_0000_0599/testlist.txt").readlines()
lines = [x.rstrip() for x in lines]

stats = np.load("tags_probs_NFNetL1V1-100-0.57141.npy")

args = np.argmax(stats, axis=0)
top1 = [lines[x] for x in args]

df = pd.read_csv("2020_0000_0599/selected_tags.csv")
df["top1"] = top1

df.to_csv("top2k.csv", index=False)
