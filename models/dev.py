import torch
import torch.nn.functional as f
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet, ResNet101_Weights, resnet101, ResNet


class DevModel(nn.Module):
    def __init__(self, backbone: ResNet, num_attrs: int, num_classes: int,
                 activation: str | None = None) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.dim = backbone.fc.in_features
        self.num_attrs, self.num_classes = num_attrs, num_classes

        self.prototype_conv = nn.Conv2d(self.dim, self.num_attrs, kernel_size=1, bias=False)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None

        self.c2y = nn.Linear(num_attrs, num_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)  # type: torch.Tensor
        b, c, h, w = features.shape

        attn_maps = self.prototype_conv(features)  # shape: [b,k,h,w]
        attr_scores = self.maxpool(attn_maps).squeeze()  # shape: [b, k]
        if self.activation:
            attr_scores = self.activation(attr_scores)
        class_scores = self.c2y(attr_scores)

        # shape: [b,num_classes], [b,k], [b,k,h,w]
        return {
            "class_scores": class_scores,
            "attr_scores": attr_scores,
            "attn_maps": attn_maps
        }

    @torch.inference_mode()
    def inference(self):
        return NotImplemented


class DevLoss(nn.Module):
    def __init__(self, attribute_weights: torch.Tensor, sigmoid: bool = False, **kwargs):
        super().__init__()
        self.l_y_coef = kwargs["l_y"]  # type: float
        self.l_c_coef = kwargs["l_c"]  # type: float
        self.l_cpt_coef = kwargs["l_cpt"]  # type: float

        self.l_y = nn.CrossEntropyLoss()
        if sigmoid:
            print("Using normal BCE")
            self.l_c = nn.BCELoss(weight=attribute_weights.to(device=kwargs["device"]))
        else:
            self.l_c = nn.BCEWithLogitsLoss(weight=attribute_weights.to(device=kwargs["device"]))


    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss_dict = {
            "l_y": self.l_y_coef * self.l_y(outputs["class_scores"], batch["class_ids"]),
            "l_c": self.l_c_coef * self.l_c(outputs["attr_scores"], batch["attr_scores"]),
            "l_cpt": self.l_cpt_coef * self.l_cpt(outputs["attn_maps"])
        }
        l_total = sum(loss_dict.values())
        return loss_dict, l_total

    @staticmethod
    def l_cpt(attn_maps: torch.Tensor):
        """Loss function for compactness of attention maps"""
        device = attn_maps.device
        b, k, h, w = attn_maps.shape
        grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        grid_w, grid_h = grid_w.to(device), grid_h.to(device)

        # Compute coordinates of max attention scores, shape: [b,k]
        _, max_attn_indices = f.max_pool2d(attn_maps.detach(), kernel_size=(h, w), return_indices=True)
        max_attn_indices = max_attn_indices.squeeze()
        max_attn_h, max_attn_w = torch.unravel_index(max_attn_indices, shape=(h, w))  # shape: [b,k], [b,k]
        max_attn_h = max_attn_h[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]
        max_attn_w = max_attn_w[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]

        attn_maps = f.sigmoid(attn_maps)  # shape: [b*k,h,w], range: [0,1]
        losses = attn_maps * ((grid_h - max_attn_h) ** 2 + (grid_w - max_attn_w) ** 2)  # shape: [b*k,h,w]

        return torch.mean(losses)

    @staticmethod
    def l_dec(prototypes: torch.Tensor, group_ids: torch.Tensor):
        """Loss function for decorrelation of attribute groups"""
        all_group_losses = []
        for i in torch.unique(group_ids):
            mask = group_ids == i
            group_loss = prototypes[mask, :].pow(2).sum().sqrt()
            all_group_losses.append(group_loss)
        return sum(all_group_losses)

