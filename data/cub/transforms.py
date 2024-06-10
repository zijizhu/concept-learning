import torchvision.transforms as t
from torchvision.models.inception import Inception_V3_Weights
from torchvision.models.resnet import ResNet101_Weights


def get_transforms_part_discovery(resolution: int = 448):
    """A set of transforms used in https://github.com/robertdvdk/part_detection."""
    train_transforms = t.Compose(
        [
            t.Resize(size=resolution, antialias=True),
            t.RandomHorizontalFlip(),
            t.ColorJitter(0.1),
            t.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            t.RandomCrop(resolution),
            t.ToTensor(),
        ]
    )
    test_transforms = t.Compose(
        [
            t.Resize(size=resolution, antialias=True),
            t.CenterCrop(size=resolution),
            t.ToTensor(),
        ]
    )

    return train_transforms, test_transforms


def get_transforms_cbm():
    """Adopted from transforms used in https://github.com/yewsiang/ConceptBottleneck for CUB dataset.
    Defined in https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py"""
    train_transforms = t.Compose([
        t.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
        t.RandomResizedCrop(299),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
    ])
    test_transforms = Inception_V3_Weights.DEFAULT.transforms()

    return train_transforms, test_transforms


def get_transforms_dev():
    train_transforms = t.Compose(
        [
            t.Resize(size=232, antialias=True),
            t.RandomHorizontalFlip(),
            t.ColorJitter(0.1),
            t.RandomCrop(224),
            t.ToTensor(),
        ]
    )
    test_transforms = ResNet101_Weights.DEFAULT.transforms()

    return train_transforms, test_transforms
