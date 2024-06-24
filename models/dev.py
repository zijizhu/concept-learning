import torch
import torch.nn.functional as f
from torch import nn
from torchvision.models import ResNet


class DevModel(nn.Module):
    def __init__(self, backbone: ResNet, num_attrs: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.dim = backbone.fc.in_features
        self.num_attrs, self.num_classes = num_attrs, num_classes

        # self.prototype_conv = nn.Conv2d(self.dim, self.num_attrs, kernel_size=1, bias=False)
        self.prototypes = nn.Parameter(2e-4 * torch.randn(num_attrs, self.dim, 1, 1))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        # self.s = nn.Sigmoid()

        self.c2y = nn.Linear(num_attrs, num_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)  # type: torch.Tensor

        # attn_maps = self.prototype_conv(features)  # shape: [b,k,h,w]
        features = f.normalize(features, p=2, dim=1)
        prototypes = f.normalize(self.prototypes, p=2, dim=1)
        attn_maps = f.conv2d(input=features, weight=prototypes)
        attn_maps = 0.5 * (attn_maps + 1)

        c = self.pool(attn_maps).squeeze(dim=(-1, -2))  # shape: [b, k]

        # c = self.s(c)
        y = self.c2y(c)

        # shape: [b,num_classes], [b,k], [b,k,h,w]
        return {
            "class_scores": y,
            "attr_scores": c,
            "attn_maps": attn_maps,
            "prototypes": torch.squeeze(self.prototypes),
        }

    @torch.inference_mode()
    def inference(self, x: torch.Tensor,
                  int_mask: torch.Tensor | None = None,
                  int_values: torch.Tensor | None = None):
        features = self.backbone(x)  # type: torch.Tensor

        features = f.normalize(features, p=2, dim=1)
        prototypes = f.normalize(self.prototypes, p=2, dim=1)
        attn_maps = f.conv2d(input=features, weight=prototypes)
        attn_maps = 0.5 * (attn_maps + 1)

        c = self.pool(attn_maps).squeeze(dim=(-1, -2))  # shape: [b, k]

        # c = self.s(c)  # shape: [b, k]

        if int_mask is not None:
            assert isinstance(int_mask, torch.Tensor) and isinstance(int_values, torch.Tensor)
            c = int_mask * int_values + (1 - int_mask) * c

        y = self.c2y(c)

        return {
            "attr_scores": c,
            "class_scores": y,
            "attn_maps": attn_maps,
            "prototypes": self.prototype_conv.weight.squeeze(),
        }


class DevLoss(nn.Module):
    def __init__(self, l_y_coef: float, l_c_coef: float, l_cpt_coef: float,
                 l_dec_coef: float, group_indices: torch.Tensor = None,
                 attribute_weights: torch.Tensor = None):
        super().__init__()
        self.l_y_coef = l_y_coef
        self.l_c_coef = l_c_coef
        self.l_cpt_coef = l_cpt_coef
        self.l_dec_coef = l_dec_coef

        self.l_y = nn.CrossEntropyLoss()
        self.l_c = nn.BCELoss(weight=attribute_weights, reduction='sum')

        self.group_indices = group_indices

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss_dict = {
            "l_y": self.l_y_coef * self.l_y(outputs["class_scores"], batch["class_ids"]),
            "l_c": self.l_c_coef * self.l_c(outputs["attr_scores"], batch["attr_scores"]),
        }
        if self.l_cpt_coef > 0:
            loss_dict["l_cpt"] = self.l_cpt_coef * self.l_cpt(outputs["attn_maps"])
        if self.l_dec_coef > 0:
            loss_dict["l_dec"] = self.l_dec_coef * self.l_dec(outputs["prototypes"])
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
        max_attn_indices = max_attn_indices.squeeze(dim=(-1, -2))
        max_attn_h, max_attn_w = torch.unravel_index(max_attn_indices, shape=(h, w))  # shape: [b,k], [b,k]
        max_attn_h = max_attn_h[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]
        max_attn_w = max_attn_w[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]

        attn_maps = f.sigmoid(attn_maps)  # shape: [b*k,h,w], range: [0,1]
        losses = attn_maps * ((grid_h - max_attn_h) ** 2 + (grid_w - max_attn_w) ** 2)  # shape: [b*k,h,w]

        return torch.mean(losses)

    def l_dec(self, prototypes: torch.Tensor):
        """Loss function for decorrelation of attribute groups"""
        group_losses = []
        for idx in self.group_indices.unique():
            mask = self.group_indices == idx
            loss = prototypes[mask, :].pow(2).sum(dim=0).add(1e-8).sum().pow(1/2.)
            group_losses.append(loss)
        return sum(group_losses)
