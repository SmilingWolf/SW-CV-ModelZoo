import json
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from tqdm import tqdm

if __name__ == "__main__":
    # open the dataframe with the tags
    # only keep general tags, no characters, no copyrights
    df = pd.read_csv("2021_0000_0899/selected_tags.csv")
    df = df[df["category"] == 0]

    with open("2021_0000_0899/testlist.txt") as f:
        samples = [x.rstrip() for x in f.readlines()]

    full_labels = np.load("2021_0000_0899/encoded_tags_test.npy")
    full_psx = np.load("tags_probs_NFNetL1V1_01_29_2022_08h20m44s.npy")

    tags_actions = {}

    # good examples: jiangshi, police, red_ascot, partially_unbuttoned
    for label_index in tqdm(df.index[::-1]):
        tag_name = df.loc[label_index]["name"]

        train_labels_with_errors = full_labels[:, label_index]
        psx = full_psx[:, label_index]

        recip = np.ones_like(psx)
        recip = recip - psx
        psx = np.stack([recip, psx], axis=1)

        ordered_label_errors = find_label_issues(
            labels=train_labels_with_errors,
            pred_probs=psx,
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
        )

        # On Danbooru a tag is more likely to be missing than to be wrong,
        # so only keep all the entries where the suggested action is "add" (argmax == 1)
        suggested_add_mask = np.argmax(psx[ordered_label_errors], axis=1) == 1
        ordered_label_errors = ordered_label_errors[suggested_add_mask]

        if len(ordered_label_errors) > 0:
            tags_actions[tag_name] = [
                int(Path(samples[x]).stem) for x in ordered_label_errors
            ]

    with open("incomplete_posts.json", "w") as f:
        f.write(json.dumps(tags_actions))
