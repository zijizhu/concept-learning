"""
Create a file containing keypoint annotations, based on:
a mapping from original parts to processed parts,
as well as a mapping from attributes to processed parts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle as pkl

from constants import PART_GROUPS

part_map_fine = {'back': "back",
                 'beak': "bill",
                 'belly': "belly",
                 'breast': "breast",
                 'crown': "crown",
                 'forehead': "forehead",
                 'left eye': "eye",
                 'left leg': "leg",
                 'left wing': "wing",
                 'nape': "nape",
                 'right eye': "eye",
                 'right leg': "leg",
                 'right wing': "wing",
                 'tail': "tail",
                 'throat': "throat"}

if __name__ == "__main__":
    parts_name_path = Path("datasets/CUB/CUB_200_2011/parts/parts.txt")
    with open(parts_name_path, "r") as fp:
        all_lines = fp.read().splitlines()
    original_part_names = [line.split(" ", 1)[1] for line in all_lines]

    part_name_map = {'back': "back",
                     'beak': "head",
                     'belly': "belly",
                     'breast': "breast",
                     'crown': "head",
                     'forehead': "head",
                     'left eye': "head",
                     'left leg': "leg",
                     'left wing': "wing",
                     'nape': "head",
                     'right eye': "head",
                     'right leg': "leg",
                     'right wing': "wing",
                     'tail': "tail",
                     'throat': "breast"}

    processed_part_names = sorted(set(part_name_map.values())) + ["others"]

    part_id_map = {}
    for i, p in enumerate(original_part_names):
        mapped_part_name = part_name_map[p]
        part_id_map[i] = processed_part_names.index(mapped_part_name)

    keypoint_ann = pd.read_csv("datasets/CUB/CUB_200_2011/parts/part_locs.txt",
                               header=None,
                               names=["image_id", "part_id", "x", "y", "visible"],
                               sep=" ")

    keypoint_ann["image_id"] -= 1
    keypoint_ann["part_id"] -= 1
    keypoint_ann = keypoint_ann[keypoint_ann["visible"] == 1]
    keypoint_ann = keypoint_ann.drop(columns=["visible"])

    keypoint_ann["part_id"] = keypoint_ann["part_id"].map(part_id_map)

    all_keypoint_anns = {}
    for img_id in keypoint_ann["image_id"].unique():
        img_keypoint_ann = keypoint_ann.loc[keypoint_ann["image_id"] == img_id, ["part_id", "x", "y"]]

        part_ids = img_keypoint_ann['part_id'].values
        keypoints = img_keypoint_ann[['x', 'y']].values
        img_keypoint_anns_processed = []
        for i, part_name in enumerate(processed_part_names):
            mask = part_ids == i
            part_keypoints = keypoints[mask]
            if part_keypoints.shape[0] > 1:
                part_keypoints = np.mean(part_keypoints, axis=0)

            if part_keypoints.shape[0] == 0:
                img_keypoint_anns_processed.append(None)
            else:
                img_keypoint_anns_processed.append(tuple(np.squeeze(part_keypoints)))
        all_keypoint_anns[img_id] = img_keypoint_anns_processed

    attr_part_map = {}
    for part_name, attr_ids in PART_GROUPS.items():
        for aid in attr_ids:
            attr_part_map[aid] = part_name

    attr_part_keypoint_anns = {
        "attribute_part_map": attr_part_map,
        "original_part_names": original_part_names,
        "processed_part_names": processed_part_names,
        "part_name_map": part_name_map,
        "part_id_map": part_id_map,
        "keypoint_annotations": all_keypoint_anns
    }

    with open("data/cub/keypoint_annotations.pkl", "wb") as fp:
        pkl.dump(attr_part_keypoint_anns, fp)
