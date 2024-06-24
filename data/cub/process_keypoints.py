"""
Create a file containing keypoint annotations, based on:
1. A mapping from original parts to processed parts,
2. A mapping from attributes to processed parts.
"""

import argparse
import pickle as pkl
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

part_map_fine = {'back': ("back",),
                 'beak': ("bill", "head",),
                 'belly': ("belly",),
                 'breast': ("breast",),
                 'crown': ("crown", "head",),
                 'forehead': ("forehead", "head",),
                 'left eye': ("eye", "head",),
                 'left leg': ("leg",),
                 'left wing': ("wing",),
                 'nape': ("nape", "head",),
                 'right eye': ("eye", "head"),
                 'right leg': ("leg",),
                 'right wing': ("wing",),
                 'tail': ("tail",),
                 'throat': ("throat",)}

part_map_coarse = {'back': ("back",),
                   'beak': ("head",),
                   'belly': ("belly",),
                   'breast': ("breast",),
                   'crown': ("head",),
                   'forehead': ("head",),
                   'left eye': ("head",),
                   'left leg': ("leg",),
                   'left wing': ("wing",),
                   'nape': ("head",),
                   'right eye': ("head",),
                   'right leg': ("leg",),
                   'right wing': ("wing",),
                   'tail': ("tail",),
                   'throat': ("breast",)}


def process_keypoints(part_map: dict[str, str],
                      keypoint_ann: pd.DataFrame,
                      original_part_names: list[str],
                      extra_part_name_list: list[str] | None = None):
    if not extra_part_name_list:
        extra_part_name_list = ["others"]
    processed_part_names = sorted(set(chain(*part_map.values()))) + extra_part_name_list

    # Create a mapping from processed part names to original part_ids
    processed_name_to_original_id = defaultdict(list)
    for original_name, mapped_processed_names in part_map_fine.items():
        original_id = original_part_names.index(original_name)
        for name in mapped_processed_names:
            processed_name_to_original_id[name].append(original_id)

    # Creat a dict of keypoint annoation with newly defined parts
    all_keypoint_anns = {}
    for img_id in tqdm(keypoint_ann["image_id"].unique()):
        img_keypoint_ann = keypoint_ann.loc[keypoint_ann["image_id"] == img_id, ["part_id", "x", "y"]]
        part_ids = img_keypoint_ann['part_id'].values
        keypoints = img_keypoint_ann[['x', 'y']].values

        img_keypoint_ann_processed = []
        for part_name in processed_part_names:
            mask = np.isin(part_ids, processed_name_to_original_id[part_name])
            part_keypoints = keypoints[mask]

            if part_keypoints.size == 0:
                img_keypoint_ann_processed.append(None)
            else:
                assert part_keypoints.ndim == 2
                part_keypoints = np.squeeze(np.mean(part_keypoints, axis=0))
                img_keypoint_ann_processed.append(tuple(part_keypoints))

        all_keypoint_anns[img_id] = img_keypoint_ann_processed

    attr_part_keypoint_anns = {
        "processed_part_names": processed_part_names,
        "keypoint_annotations": all_keypoint_anns
    }

    return attr_part_keypoint_anns


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to process keypoint annotations")
    parser.add_argument("--dataset_dir", type=str, default="datasets/CUB")
    parser.add_argument("--part_granularity", type=str, nargs="+",
                        choices=["coarse", "fine"], default=["coarse", "fine"])
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / "CUB_200_2011"
    assert dataset_dir.exists()

    with open(dataset_dir / "parts" / "parts.txt") as fp:
        all_lines = fp.read().splitlines()
    original_part_names = [line.split(" ", 1)[1] for line in all_lines]

    keypoint_ann = pd.read_csv(dataset_dir / "parts" / "part_locs.txt", header=None,
                               names=["image_id", "part_id", "x", "y", "visible"], sep=" ")
    keypoint_ann["image_id"] -= 1
    keypoint_ann["part_id"] -= 1

    keypoint_ann = keypoint_ann[keypoint_ann["visible"] == 1]
    keypoint_ann = keypoint_ann.drop(columns=["visible"])

    if "fine" in args.part_granularity:
        print("Processing annotations using fine-grained part definition...")
        processed_keypoint_ann_fine = process_keypoints(part_map_fine, keypoint_ann,
                                                        original_part_names, ["others"])
        
        # Create a mappaing from attrbutes to processed part names
        attribute_df = pd.read_csv(dataset_dir / "attributes" / "attributes.txt", header=None,
                                   names=["attribute_id", "attribute"],
                                   sep=" ").drop(columns=["attribute_id"])
        processed_part_names = processed_keypoint_ann_fine["processed_part_names"]

        def map_attr_to_part(attr: str):
            for p in processed_part_names:
                if p in attr:
                    return p
            return "others"

        attribute_df["part"] = attribute_df["attribute"].map(map_attr_to_part)

        processed_keypoint_ann_fine["attribute_part_map"] = attribute_df["part"].to_list()

        # Save processed annotation dict as pkl
        ann_save_path = dataset_dir / "keypoint_annotations_fine.pkl"
        with open(ann_save_path, "wb") as fp:
            pkl.dump(processed_keypoint_ann_fine, fp)
        print(f"Saved processed annotations to {ann_save_path.as_posix()}")
    
    if "coarse" in args.part_granularity:
        print("Processing annotations using coarse-grained part definition...")
        processed_keypoint_ann_coarse = process_keypoints(part_map_coarse, keypoint_ann,
                                                          original_part_names, ["others"])
        
        # Load the mappaing from attrbutes to processed part names from txt
        with open(Path(__file__).parent / "attribute_part_map_coarse.txt") as fp:
            attribute_part_map = fp.read().splitlines()
        processed_keypoint_ann_coarse["attribute_part_map"] = attribute_part_map

        # Save processed annotation dict as pkl
        ann_save_path = dataset_dir / "keypoint_annotations_coarse.pkl"
        with open(ann_save_path, "wb") as fp:
            pkl.dump(processed_keypoint_ann_coarse, fp)
        print(f"Saved processed annotations to {ann_save_path.as_posix()}")
