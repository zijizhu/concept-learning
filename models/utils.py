import torch
import torch.nn.functional as f
from torch import nn
from torch import optim
import timm


class Backbone(nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        if name == 'inception_v3':
            self.backbone = timm.create_model('inception_v3', pretrained=True, aux_logits=False)
        else:
            raise NotImplementedError
        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, batch):
        x = batch['pixel_values']
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)
        x = self.backbone.head_drop(x)
        y = self.fc(x)
        return {'class_preds': y}


def loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
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
