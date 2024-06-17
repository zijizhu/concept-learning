"""
Script to loop through image folders,
crop each image according to the bbox annotation.
The crop size is a square with size equal to the larger side of annotated bbox.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to crop images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / "CUB_200_2011"

    bbox_df = pd.read_csv(dataset_dir / "bounding_boxes.txt",
                          header=None, names=["image_id", "x", "y", "w", "h"], sep=" ")
    main_df = pd.read_csv(dataset_dir / "images.txt",
                          sep=" ", header=None, names=["image_id", "filename"])
    merged_df = pd.merge(main_df, bbox_df, how="left", on="image_id")
    merged_df["image_id"] -= 1
    merged_df = merged_df.set_index("image_id", drop=True)

    for directory in tqdm((dataset_dir / "images").glob("*/")):
        out_dir = directory / "cropped"
        out_dir.mkdir(exist_ok=True)
        for fn in directory.glob("*.jpg"):
            image = Image.open(fn).convert("RGB")

            filename = f"{directory.name}/{fn.name}"

            bbox = merged_df.loc[merged_df["filename"] == filename, ["x", "y", "w", "h"]].values
            x, y, w, h = np.squeeze(bbox)

            cx, cy = x + w / 2, y + h / 2
            new_w = new_h = max(w, h) + 20
            new_x, new_y = max(cx - new_w / 2, 0), max(cy - new_h / 2, 0)

            image_cropped = image.crop((new_x, new_y, new_x + new_w, new_y + new_h,))
            image_cropped.save(out_dir / fn.name)
