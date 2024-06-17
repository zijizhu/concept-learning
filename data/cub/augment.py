"""
Migrated from ProtoPNet
"""
import Augmentor
import os

import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse


def get_augmented_train_df(image_dir: str | Path, train_df: pd.DataFrame) -> pd.DataFrame:
    image_dir = Path(image_dir)
    augmented_filenames = []
    augmented_class_ids = []
    for class_image_dir in image_dir.glob("*/"):
        class_train_mask = train_df["filename"].str.split("/").str[0] == class_image_dir.name
        class_id = train_df.loc[class_train_mask, "class_id"].unique()[0]
        original_filenames = train_df.loc[class_train_mask, "filename"].str.split("/").str[-1]
        for augmented_fn in (class_image_dir / "cropped" / "augmented").glob("*"):
            if any(fn in augmented_fn.name for fn in original_filenames):
                augmented_filenames.append(f"{class_image_dir.name}/cropped/augmented/{augmented_fn.name}")
                augmented_class_ids.append(class_id)

    result = pd.DataFrame(
        {'filename': augmented_filenames, 'class_id': augmented_class_ids, "is_train": [1] * len(augmented_filenames)})
    result.index.name = "image_id"

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script to crop images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / "CUB_200_2011"

    for directory in (dataset_dir / "images").glob("*/"):
        in_dir = directory / "cropped"
        assert len(list(in_dir.glob("*"))) > 0

        out_dir = directory / "cropped" / "augmented"
        out_dir.mkdir(exist_ok=True)

        # # Rotation
        p = Augmentor.Pipeline(source_directory=str(in_dir), output_directory=out_dir.name)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p

        # Skew
        p = Augmentor.Pipeline(source_directory=str(in_dir), output_directory=out_dir.name)
        p.skew(probability=1, magnitude=0.2)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p

        # Shear
        p = Augmentor.Pipeline(source_directory=str(in_dir), output_directory=out_dir.name)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p

        # Random Distortion
        p = Augmentor.Pipeline(source_directory=str(in_dir), output_directory=out_dir.name)
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p
        
        print(f"Processed {str(in_dir)}")
