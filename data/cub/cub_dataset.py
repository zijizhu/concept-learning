import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    def __init__(self,
                 dataset_dir: str | Path,
                 split: str = "train",
                 use_attrs: str | Path | np.ndarray | torch.Tensor = "binary",
                 use_attr_mask: str | Path | np.ndarray = None,
                 use_splits: str| Path | dict | None = None,
                 transforms: t.Compose | None = None) -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir

        #####################################################
        # Load dataframes that store information of samples #
        #####################################################

        file_paths_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "images.txt"),
            sep=" ", header=None, names=["image_id", "filename"])
        image_labels_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ", header=None, names=["image_id", "class_id"])
        train_test_split_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "train_test_split.txt"),
            sep=" ", header=None, names=["image_id", "is_train"])

        main_df = (file_paths_df
                   .merge(image_labels_df, on="image_id")
                   .merge(train_test_split_df, on="image_id"))

        main_df["image_id"] -= 1
        main_df["class_id"] -= 1

        self.main_df = main_df.set_index("image_id")

        if not use_splits:
            train_val_mask = main_df["is_train"] == 1
            test_mask = ~train_val_mask
            train_val_image_ids = main_df.loc[train_val_mask, "image_id"].unique()
            test_image_ids = main_df.loc[test_mask, "image_id"].unique()

            train_image_ids, val_image_ids = train_test_split(train_val_image_ids, test_size=0.2)
            self.image_ids = {"train": train_image_ids,
                              "val": val_image_ids,
                              "test": test_image_ids}
        else:
            if isinstance(use_splits, str) or isinstance(use_splits, Path):
                use_splits = np.load(use_splits)
            self.image_ids = {
                "train": use_splits["train"],
                "val": use_splits["val"],
                "test": use_splits["test"]
            }
            self.image_ids['train_val'] = np.concatenate([self.image_ids['train'], self.image_ids['val']])

        ###############################
        # Load and process attributes #
        ###############################

        # Load attribute names
        attr_df = pd.read_csv(
            os.path.join(dataset_dir, "attributes.txt"), sep=' ', usecols=[1],
            header=None, names=['attribute'])
        attr_df['attribute'] = attr_df['attribute'].str.replace('_', ' ', regex=True)
        attr_df[['attribute_group', 'value']] = attr_df['attribute'].str.split('::', n=1, expand=True)

        attr_mask = None
        if use_attr_mask:
            if isinstance(use_attr_mask, str) or isinstance(use_attr_mask, Path):
                attr_mask = np.loadtxt(use_attr_mask).astype(bool)
            elif isinstance(use_attr_mask, np.ndarray):
                attr_mask = use_attr_mask
            else:
                raise NotImplementedError

            attr_df = attr_df[attr_mask]

        self.attributes_df = attr_df.reset_index(drop=True)

        # Load attribute vectors
        if use_attrs in ["binary", "continuous"]:
            attr_vectors = np.loadtxt(
                os.path.join(dataset_dir, "CUB_200_2011", "attributes",
                             "class_attribute_labels_continuous.txt"))
            attr_vectors /= 100

            if attr_mask:
                attr_vectors = attr_vectors[:, attr_mask]

            if use_attrs == "binary":
                attr_vectors = np.where(attr_vectors >= 0.5, 1, 0)
        else:
            if isinstance(use_attrs, str) or isinstance(use_attrs, Path):
                attr_vectors = np.loadtxt(use_attrs)
            elif isinstance(use_attrs, np.ndarray) or isinstance(use_attrs, torch.Tensor):
                attr_vectors = use_attrs
            else:
                raise NotImplementedError

        self.attribute_vectors = attr_vectors
        assert len(self.attributes_df) == self.attribute_vectors.shape[1]

        #####################################################
        # Load and process class names in a readable format #
        #####################################################

        class_names_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "classes.txt"), sep=" ",
            header=None, names=["class_id", "class_name"])

        self.class_names = (class_names_df["class_name"].str.split(".").str[-1]
                            .replace("_", " ", regex=True).to_list())

        #################################
        # Load part or attribute groups #
        #################################
        self.group_names = sorted(self.attributes_df['attribute_group'].unique())
        self.attribute_group_indices = (self.attributes_df['attribute_group']
                                        .map(self.group_names.index)
                                        .to_numpy())

        if not transforms:
            transforms = t.ToTensor()
        self.transforms = transforms

    @property
    def attribute_weights(self):
        """Compute attribute class weights following Concept Bottleneck Models"""
        train_image_ids = self.image_ids["train"]
        train_class_ids = self.main_df.iloc[train_image_ids]["class_id"].to_numpy()
        return len(train_image_ids) / np.sum(self.attribute_vectors[train_class_ids, :], axis=0) - 1

    @property
    def attribute_group_indices_pt(self):
        return torch.tensor(self.attribute_group_indices, dtype=torch.long)

    @property
    def attribute_vectors_pt(self):
        """Attribute vector for each class in type torch.Tensor"""
        return torch.tensor(self.attribute_vectors, dtype=torch.float32)

    def __len__(self):
        return len(self.image_ids[self.split])

    def __getitem__(self, idx):
        image_id = self.image_ids[self.split][idx]

        filename, class_id, _ = self.main_df.iloc[image_id]

        path_to_image = os.path.join(self.dataset_dir, "CUB_200_2011", "images", filename)
        image = Image.open(path_to_image).convert("RGB")

        attr_scores = self.attribute_vectors_pt[class_id]

        pixel_values = self.transforms(image)

        return {
            "image_ids": torch.tensor(image_id, dtype=torch.long),
            "pixel_values": pixel_values,
            "class_ids": torch.tensor(class_id, dtype=torch.long),
            "attr_scores": attr_scores.clone(),
        }
