import numpy as np
import pandas as pd

lines = open("2021_0000_0899/testlist.txt").readlines()
lines = [x.rstrip() for x in lines]

stats = np.load("tags_probs_NFNetL1V1_01_29_2022_08h20m44s.npy")

args = np.argmax(stats, axis=0)
top1 = [lines[x] for x in args]

df = pd.read_csv("2021_0000_0899/selected_tags.csv")
df["top1"] = top1

df.to_csv("top2k.csv", index=False)
