import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class CUBDataset(Dataset):
    def __init__(self, dataset_dir, split='train', transforms=None) -> None:
        super().__init__()
        self.split = split
        self.transforms = transforms
        self.dataset_dir = dataset_dir
        file_path_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'images.txt'), sep=' ',
                                   header=None, names=['img_id', 'file_path'])
        img_class_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'image_class_labels.txt'),
                                   sep=' ', header=None, names=['img_id', 'class_id'])
        train_test_split_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'train_test_split.txt'),
                                          sep=' ', header=None, names=['img_id', 'is_train'])
        merged_df = file_path_df.merge(img_class_df, on='img_id').merge(train_test_split_df, on='img_id')
        
        # Make class_id 0-indexed
        merged_df['class_id'] = merged_df['class_id'] - 1
        self.class_id2name = {}
        for line in open('datasets/CUB_200_2011/classes.txt'):
            [class_id, class_name] = line.strip().split(' ')
            self.class_id2name[int(class_id) - 1] = class_name

        train_df = merged_df[merged_df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
        test_df = merged_df[merged_df['is_train'] == 0].drop(columns=['is_train']).reset_index(drop=True)

        self.annotations = {'train': train_df, 'test': test_df}

    def __len__(self):
        return len(self.annotations[self.split])

    def __getitem__(self, idx):
        img_id, file_path, class_id = self.annotations[self.split].iloc[idx]
        image = Image.open(os.path.join(self.dataset_dir, 'CUB_200_2011', 'images', file_path))
        if self.transforms is not None:
            image = self.transforms(image)
        return torch.tensor(img_id), image, torch.tensor(class_id)
