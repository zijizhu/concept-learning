import logging
from pathlib import Path

import albumentations as A
import numpy as np
import torch

from ..data.cub.cub_dataset import CUBDataset
from .loc import loc_eval


class FakeModel:
    def __init__(self, keypoint_ann, original_image_sizes) -> None:
        self.ann = keypoint_ann
        self.transform = A.Compose(
            [A.Resize(232, 232), A.CenterCrop(224, 224)],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"], remove_invisible=True
            ),
        )
        self.idx = 0
        self.original_image_sizes = original_image_sizes

    def __call__(self, img):
        im_id, img_size = self.original_image_sizes[self.idx]
        im_id = im_id.item()
        im = np.zeros(tuple(img_size)).transpose(1, 2, 0)
        sample_kp_ann = self.ann["keypoint_annotations"][im_id]
        kp_labels = self.ann["processed_part_names"]
        keypoints, keypoint_labels = [], []
        for kp, label in zip(sample_kp_ann, kp_labels):
            if kp is not None:
                keypoints.append(kp)
                keypoint_labels.append(label)
        transformed = self.transform(image=im, keypoints=keypoints, class_labels=keypoint_labels)
        keypoint_interpolated = []
        for kp in transformed["keypoints"]:
            x, y = kp
            keypoint_interpolated.append((int(x // 32), int(y // 32),))
        attn_map = torch.zeros((312, 7, 7,), dtype=torch.float32)
        for i in range(312):
            part = self.ann["attribute_part_map"][i]
            if part in keypoint_labels:
                part_idx = keypoint_labels.index(part)
                x, y = keypoint_interpolated[part_idx]
                attn_map[i, y, x] = 10.0
        self.idx += 1
        return {"attn_maps": attn_map}


if __name__ == "__main__":
    dataset_test = CUBDataset(
        Path("datasets") / "CUB",
        split="test",
        use_augmentation=None,
        use_attrs="binary",
        use_attr_mask=False,
        use_part_group="fine",
        use_splits=None,
        transforms=None,
    )

    original_image_sizes = []
    for sample in dataset_test:
        original_image_sizes.append((sample["image_ids"], sample["pixel_values"].shape,))

    model = FakeModel(dataset_test.part_keypoint_annotation, original_image_sizes=original_image_sizes)
    loc_eval(
        model=model,
        dataset_test=dataset_test,
        output_dir=Path("./"),
        logger=logging.getLogger(),
        cropped=False,
    )
