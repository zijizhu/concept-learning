import torch
from torch import nn
from torch import optim
from torchvision.models import inception_v3, Inception_V3_Weights, InceptionOutputs
import timm
from .utils import Backbone


class CBM(nn.Module):
    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.backbone = backbone
        self.f2c = nn.Linear(backbone.fc.in_features, num_concepts)
        self.c2y = nn.Linear(num_concepts, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]):
        x = batch['pixel_values']
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)
        f = self.backbone.head_drop(x)

        c = self.f2c(f)
        y = self.c2y(c)
        return {
            "concept_preds": c,
            "class_preds": y
        }

    @torch.inference_mode()
    def inference(self, batch: dict[str, torch.Tensor], int_mask: torch.Tensor | None = None,
                  int_values: torch.Tensor | None = None):
        x = batch['pixel_values']
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)
        f = self.backbone.head_drop(x)
        c = self.f2c(f)

        if int_mask is not None:
            assert isinstance(int_mask, torch.Tensor) and isinstance(int_values, torch.Tensor)
            c = int_mask * int_values + (1 - int_mask) * c

        y = self.c2y(c)

        return {
            "concept_preds": c,
            "class_preds": y
        }


class CBMLoss(nn.Module):
    def __init__(self, loss_coefs: dict[str, float]):
        super().__init__()
        self.l_c_coef = loss_coefs['l_c']
        self.l_y_coef = loss_coefs['l_y']
        self.l_c = nn.BCEWithLogitsLoss()
        self.l_y = nn.CrossEntropyLoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        concept_labels, class_labels = batch['attr_scores'], batch['class_ids']
        concept_preds, class_preds = outputs['concept_preds'], outputs['class_preds']
        loss_dict = {
            "l_c": self.l_c_coef * self.l_c(concept_preds, concept_labels),
            "l_y": self.l_y_coef * self.l_y(class_preds, class_labels)
        }
        l_total = sum(loss_dict.values())
        return loss_dict, l_total


def load_cbm_for_training(
    backbone_name: str,
    checkpoint_path: str,
    num_classes: int,
    num_concepts: int,
    loss_coef_dict: dict[str, float],
    lr: float,
    weight_decay: float
):
    if backbone_name == 'inception_v3':
        backbone = timm.create_model('inception_v3', pretrained=True, aux_logits=False)
        state_dict = torch.load(checkpoint_path)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        state_dict = {k.split('.', 1)[-1]: v for k, v in state_dict.items()}
    else:
        raise NotImplementedError
    backbone.load_state_dict(state_dict)
    net = CBM(
        backbone=backbone,
        num_concepts=num_concepts,
        num_classes=num_classes
    )
    loss = CBMLoss(loss_coefs=loss_coef_dict)
    optimizer = optim.SGD(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    return net, loss, optimizer, None
