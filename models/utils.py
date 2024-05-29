import torch
import torch.nn.functional as f
from torch import nn
from torch import optim
from torchvision.models import inception_v3, Inception_V3_Weights, InceptionOutputs


class Backbone(nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        if name == 'inception_v3':
            self.backbone = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, batch):
        x = batch['pixel_values']
        y = self.backbone(x)
        if isinstance(y, InceptionOutputs):
            y = y.logits
        return {'class_preds': y}

    def state_dict(self, *args, **kwargs):
        return self.backbone.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.backbone.load_state_dict(*args, **kwargs)


def loss_fn(outputs: dict[str, torch.Tensor | InceptionOutputs], batch: dict[str, torch.Tensor]):
    preds = outputs['class_preds']

    loss = f.cross_entropy(preds, batch['class_ids'])
    return {'l_total': loss.item()}, loss


def load_backbone_for_finetuning(
    backbone_name: str,
    num_classes: int,
    lr: float,
    weight_decay: float
):
    net = Backbone(name=backbone_name, num_classes=num_classes)
    optimizer = optim.SGD(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    return net, loss_fn, optimizer, None
