import torch
import torch.nn.functional as F
from torch import nn


class DINOClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.fc_c = nn.Linear(768, 112)
        self.s = nn.Sigmoid()
        self.fc_y = nn.Linear(112, 200)
    
    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        c = self.fc_c(features)
        c_probs = self.s(c)
        y = self.fc_y(c_probs)

        return {"attr_preds": c, "class_preds": y}
    
    @torch.inference_mode()
    def inference(self,
                  x: torch.Tensor,
                  int_mask: torch.Tensor | None = None,
                  int_values: torch.Tensor | None = None):
        features = self.backbone(x)
        c = self.fc_c(features)
        c_probs = self.s(c)

        if int_mask is not None:
            assert isinstance(int_mask, torch.Tensor) and isinstance(int_values, torch.Tensor)
            c_probs = int_mask * int_values + (1 - int_mask) * c_probs

        y = self.fc_y(c_probs)

        return {"attr_preds": c, "class_preds": y}
    

class DINOLocClassifier(nn.Module):
    def __init__(self, num_attrs: int) -> None:
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.prototypes = nn.Parameter(1e-4 * torch.randn(num_attrs, 768))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.s = nn.Sigmoid()
        self.fc_y = nn.Linear(num_attrs, 200)
    
    def forward(self, x: torch.Tensor):
        feature_dict = self.backbone.forward_features(x)
        attn_maps = feature_dict["x_norm_patchtokens"] @ self.prototypes.T  # shape: [b, n_patches, n_attrs]
        b, n_patches, n_attrs = attn_maps.shape
        attn_maps = attn_maps.permute(0, 2, 1)
        c = self.pool(attn_maps).squeeze()
        c_probs = self.s(c)
        y = self.fc_y(c_probs)

        return {"attr_preds": c, "class_preds": y, "attn_maps": attn_maps.view(b, n_attrs, 16, 16)}
    
    @torch.inference_mode()
    def inference(self,
                  x: torch.Tensor,
                  int_mask: torch.Tensor | None = None,
                  int_values: torch.Tensor | None = None):
        feature_dict = self.backbone.forward_features(x)
        attn_maps = feature_dict["x_norm_patchtokens"] @ self.prototypes.T  # shape: [b, n_patches, n_attrs]
        b, n_patches, n_attrs = attn_maps.shape
        attn_maps = attn_maps.permute(0, 2, 1)
        c = self.pool(attn_maps).squeeze()
        c_probs = self.s(c)

        if int_mask is not None:
            assert isinstance(int_mask, torch.Tensor) and isinstance(int_values, torch.Tensor)
            c_probs = int_mask * int_values + (1 - int_mask) * c_probs

        y = self.fc_y(c_probs)

        return {"attr_preds": c, "class_preds": y, "attn_maps": attn_maps.view(b, n_attrs, 16, 16)}


class CBMCriterion(nn.Module):
    def __init__(self, l_y_coef: float, l_c_coef: float, l_cpt_coef: float, l_dec_coef: float,
                 group_indices: torch.Tensor) -> None:
        super().__init__()
        self.l_y_coef = l_y_coef
        self.l_c_coef = l_c_coef
        self.l_cpt_coef = l_cpt_coef
        self.l_dec_coef = l_dec_coef
        self.xe = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.group_indices = group_indices
    
    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                prototype_weights: torch.Tensor):
        loss_dict = {
            "l_y": self.l_y_coef * self.xe(outputs["class_preds"], batch["class_ids"]),
            "l_c": self.l_c_coef * self.bce(outputs["attr_preds"], batch["attr_scores"])
        }
        if self.l_cpt_coef > 0:
            loss_dict["l_cpt"] = self.l_cpt_coef * self.l_cpt(outputs["attn_maps"])
        if self.l_dec_coef > 0:
            loss_dict["l_dec"] = self.l_dec_coef * self.l_dec(prototype_weights)
        return loss_dict
    

    @staticmethod
    def l_cpt(attn_maps: torch.Tensor):
        """Loss function for compactness of attention maps"""
        device = attn_maps.device
        b, k, h, w = attn_maps.shape
        grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        grid_w, grid_h = grid_w.to(device), grid_h.to(device)

        # Compute coordinates of max attention scores
        _, max_attn_indices = F.max_pool2d(attn_maps.detach(), kernel_size=(h, w,), return_indices=True)
        max_attn_indices = max_attn_indices.squeeze(dim=(-1, -2,))  # shape: [b,k]
        max_attn_h, max_attn_w = torch.unravel_index(max_attn_indices, shape=(h, w,))  # shape: [b,k], [b,k]
        max_attn_h = max_attn_h[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]
        max_attn_w = max_attn_w[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]

        attn_maps = F.sigmoid(attn_maps)  # shape: [b*k,h,w], range: [0,1]
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
