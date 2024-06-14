import os
from pathlib import Path

import Augmentor
import pandas as pd


def run_augmentation(image_dir: str | Path):
    ...


def get_augmented_image_info(dataset_dir: str | Path):
    image_dir = Path(dataset_dir) / "CUB" / "CUB_200_2011" / "images"
    all_class_ids, all_aug_file_paths = [], []
    for class_image_dir in image_dir.glob("*/"):
        class_id = int(class_image_dir.name.split(".")[0]) - 1
        if not (class_image_dir / "augmented").is_dir():
            run_augmentation(class_image_dir)
        aug_filenames = [fn.name for fn in (class_image_dir / "augmented").glob("*")]
        aug_file_paths = [os.path.join(class_image_dir.name, fn) for fn in aug_filenames]
        all_aug_file_paths += aug_file_paths
        all_class_ids += [class_id] * len(aug_file_paths)

    return pd.DataFrame({"class_id": all_class_ids,
                         "filename": all_aug_file_paths,
                         "is_train": [1] * len(all_class_ids)})


if __name__ == "__main__":
    get_augmented_image_info(dataset_dir=Path("datasets/"))
