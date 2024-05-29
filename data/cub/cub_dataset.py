import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t
import torchvision.transforms.functional as f
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .constants import PART_GROUPS, SELECTED_CONCEPTS


class CUBDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            use_attrs: str = "binary",
            num_attrs: int = 112,
            split: str = "train",
            groups: str = "attributes",
            transforms: t.Compose | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir

        #####################################################
        # Load dataframes that store information of samples #
        #####################################################

        file_paths_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "images.txt"),
            sep=" ",
            header=None,
            names=["image_id", "filename"],
        )
        image_labels_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            header=None,
            names=["image_id", "class_id"],
        )
        train_test_split_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            header=None,
            names=["image_id", "is_train"],
        )

        main_df = file_paths_df.merge(image_labels_df, on="image_id").merge(
            train_test_split_df, on="image_id"
        )

        main_df["image_id"] -= 1
        main_df["class_id"] -= 1

        train_val_mask = main_df["is_train"] == 1
        test_mask = ~train_val_mask
        train_val_image_ids = main_df.loc[train_val_mask, "image_id"].unique()  # type: np.ndarray
        test_image_ids = main_df.loc[test_mask, "image_id"].unique()  # type: np.ndarray

        train_image_ids, val_image_ids = train_test_split(train_val_image_ids, test_size=0.2)

        self.main_df = main_df.set_index("image_id")
        self.image_ids = {"train": train_image_ids,
                          "val": val_image_ids,
                          "test": test_image_ids}

        ###############################
        # Load and process attributes #
        ###############################

        # Load attribute names
        attr_df = pd.read_csv(
            os.path.join(dataset_dir, "attributes.txt"),
            sep=' ',
            usecols=[1],
            header=None,
            names=['attribute']
        )
        attr_df['attribute'] = attr_df['attribute'].str.replace('_', ' ', regex=True)
        attr_df[['attribute_group', 'value']] = attr_df['attribute'].str.split('::', n=1, expand=True)

        if num_attrs == 112:
            attr_df = attr_df.iloc[SELECTED_CONCEPTS]

        self.attributes_df = attr_df.reset_index(drop=True)

        # Load attribute vectors
        assert use_attrs in ["binary", "continuous", "cem"]
        if use_attrs == "cem":
            attr_vectors = np.loadtxt(
                os.path.join(
                    os.path.dirname(__file__),
                    "cem_attributes.txt"
                )
            )

        else:
            assert num_attrs in [112, 312]
            attr_vectors = np.loadtxt(
                os.path.join(
                    dataset_dir,
                    "CUB_200_2011",
                    "attributes",
                    "class_attribute_labels_continuous.txt",
                )
            )
            if num_attrs == 112:
                attr_vectors = attr_vectors[:, SELECTED_CONCEPTS]
                attr_vectors /= 100

            if use_attrs == "binary":
                attr_vectors = np.where(attr_vectors >= 0.5, 1, 0)

        self.use_attributes = use_attrs
        self.attributes_vectors = attr_vectors

        #####################################################
        # Load and process class names in a readable format #
        #####################################################

        class_names_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "classes.txt"),
            sep=" ",
            header=None,
            names=["class_id", "class_name"],
        )

        self.class_names = (
            class_names_df["class_name"]
            .str
            .split(".")
            .str[-1]
            .replace("_", " ", regex=True)
            .to_list()
        )

        #################################
        # Load part or attribute groups #
        #################################
        assert groups in ["parts", "attributes"]
        self.groups = groups
        if groups == 'parts':
            assert num_attrs == 312
            self.group_names = sorted(PART_GROUPS.keys())
            group_indices = np.zeros(312, dtype=int)
            for i, name in enumerate(self.group_names):
                attr_indices = PART_GROUPS[name]
                group_indices[attr_indices] = i
            self.attribute_group_indices = group_indices
        elif groups == 'attributes':
            self.group_names = sorted(self.attributes_df['attribute_group'].unique())
            self.attribute_group_indices = (self.attributes_df['attribute_group']
                                            .map(self.group_names.index)
                                            .to_numpy())
        else:
            raise NotImplementedError

        self.transforms = transforms

    @property
    def attribute_group_indices_pt(self):
        return torch.tensor(self.attribute_group_indices, dtype=torch.long)

    @property
    def attribute_vectors_pt(self):
        """Attribute vector for each class in type torch.Tensor"""
        return torch.tensor(self.attributes_vectors, dtype=torch.float32)

    def visualize_random_classes(self, grid_size: tuple[int, int] = (5, 5)):
        """Sample one image from different classes of the dataset and visualize them with matplotlib.
        """
        transforms = t.Compose([
            t.Resize(size=232, interpolation=t.InterpolationMode.BILINEAR),
            t.CenterCrop(size=224)
        ])
        nrows, ncols = grid_size
        nsamples = nrows * ncols
        assert nsamples <= len(self.class_names)

        # Sample class ids to visualize
        split_sample_indices = self.image_ids[self.split]
        split_df = self.main_df.iloc[split_sample_indices]  # type: pd.DataFrame
        sampled_class_ids = np.random.randint(0, len(self.class_names), size=nsamples)

        # For each sampled class id, sample one image id belongs to this class
        sampled_image_ids = []
        for class_id in sampled_class_ids:
            mask = split_df['class_id'] == class_id
            class_image_ids = split_df.loc[mask].index.to_numpy()
            sampled = np.random.choice(class_image_ids, size=1)
            sampled_image_ids.append(sampled)
        sampled_image_ids = np.concatenate(sampled_image_ids)

        # Maka a figure and visualize
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * nrows, 3 * ncols))
        fig.tight_layout()
        for ax, class_id, image_id in zip(axes.flat, sampled_class_ids, sampled_image_ids):
            filename = self.main_df.loc[image_id, 'filename']
            image_path = os.path.join(self.dataset_dir, "CUB_200_2011", "images", filename)
            image = transforms(Image.open(image_path).convert('RGB'))
            ax.imshow(image)
            ax.set_title(f'Class id :{class_id}\n{self.class_names[class_id]}')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        return fig

    def visualize_random_class_samples(self, class_id: int, grid_size: tuple[int] = (5, 5)):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_ids[self.split])

    def __getitem__(self, idx):
        image_id = self.image_ids[self.split][idx]

        filename, class_id, _ = self.main_df.iloc[image_id]

        path_to_image = os.path.join(self.dataset_dir, "CUB_200_2011", "images", filename)
        image = Image.open(path_to_image).convert("RGB")

        attr_scores = self.attribute_vectors_pt[class_id]

        pixel_values = self.transforms(image) if self.transforms else f.pil_to_tensor(image)

        return {
            "image_ids": image_id,
            "pixel_values": pixel_values,
            "class_ids": torch.tensor(class_id, dtype=torch.long),
            "attr_scores": attr_scores.clone(),
        }
