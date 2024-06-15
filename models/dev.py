import torch
import torch.nn.functional as f
from torch import nn
from torchvision.models import ResNet


class DevModel(nn.Module):
    def __init__(self, backbone: ResNet, num_attrs: int, num_classes: int,
                 use_attention: bool = False) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.dim = backbone.fc.in_features
        self.num_attrs, self.num_classes = num_attrs, num_classes
        self.use_attention = use_attention

        self.prototype_conv = nn.Conv2d(self.dim, self.num_attrs, kernel_size=1, bias=False)
        if use_attention:
            print("Model uses attention")
            self.pool = nn.MultiheadAttention(embed_dim=7*7, num_heads=1, dropout=0.1, batch_first=True)
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.s = nn.Sigmoid()

        self.c2y = nn.Linear(num_attrs, num_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)  # type: torch.Tensor
        b, c, h, w = features.shape

        attn_maps = self.prototype_conv(features)  # shape: [b,k,h,w]
        if self.use_attention:
            attn_maps = attn_maps.view(b, self.num_attrs, -1)
            attn_maps, _ = self.pool(query=attn_maps, key=attn_maps, value=attn_maps)
            c = f.adaptive_avg_pool1d(attn_maps, 1).squeeze(dim=-1)
            attn_maps = attn_maps.view(b, self.num_attrs, h, w)
        else:
            c = self.pool(attn_maps).squeeze(dim=(-1, -2))  # shape: [b, k]

        c = self.s(c)
        y = self.c2y(c)

        # shape: [b,num_classes], [b,k], [b,k,h,w]
        return {
            "class_scores": y,
            "attr_scores": c,
            "attn_maps": attn_maps
        }

    @torch.inference_mode()
    def inference(self, x: torch.Tensor,
                  int_mask: torch.Tensor | None = None,
                  int_values: torch.Tensor | None = None):
        features = self.backbone(x)  # type: torch.Tensor
        b, c, h, w = features.shape
        attn_maps = self.prototype_conv(features)  # shape: [b,k,h,w]

        if self.use_attention:
            attn_maps = attn_maps.view(b, self.num_attrs, -1)
            attn_maps, _ = self.pool(query=attn_maps, key=attn_maps, value=attn_maps)
            c = f.adaptive_avg_pool1d(attn_maps, 1).squeeze(dim=-1)
            attn_maps = attn_maps.view(b, self.num_attrs, h, w)
        else:
            c = self.pool(attn_maps).squeeze(dim=(-1, -2))  # shape: [b, k]
        if self.s:
            c = self.s(c)  # shape: [b, k]

        if int_mask is not None:
            assert isinstance(int_mask, torch.Tensor) and isinstance(int_values, torch.Tensor)
            c = int_mask * int_values + (1 - int_mask) * c

        y = self.c2y(c)

        return {
            "attr_scores": c,
            "class_scores": y,
            "attn_maps": attn_maps
        }


class DevLoss(nn.Module):
    def __init__(self, l_y_coef: float, l_c_coef: float, l_cpt_coef: float,
                 attribute_weights: torch.Tensor = None):
        super().__init__()
        self.l_y_coef = l_y_coef
        self.l_c_coef = l_c_coef
        self.l_cpt_coef = l_cpt_coef

        self.l_y = nn.CrossEntropyLoss()
        self.l_c = nn.BCELoss(weight=attribute_weights, reduction='sum')

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss_dict = {
            "l_y": self.l_y_coef * self.l_y(outputs["class_scores"], batch["class_ids"]),
            "l_c": self.l_c_coef * self.l_c(outputs["attr_scores"], batch["attr_scores"]),
        }
        if self.l_cpt_coef > 0:
            loss_dict["l_cpt"] = self.l_cpt_coef * self.l_cpt(outputs["attn_maps"])
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

    @staticmethod
    def l_dec(prototypes: torch.Tensor, group_ids: torch.Tensor):
        """Loss function for decorrelation of attribute groups"""
        all_group_losses = []
        for i in torch.unique(group_ids):
            mask = group_ids == i
            group_loss = prototypes[mask, :].pow(2).sum().sqrt()
            all_group_losses.append(group_loss)
        return sum(all_group_losses)
