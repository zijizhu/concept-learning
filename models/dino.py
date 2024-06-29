import torch
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


class CBMCriterion(nn.Module):
    def __init__(self, l_y_coef: float, l_c_coef: float) -> None:
        super().__init__()
        self.l_y_coef = l_y_coef
        self.l_c_coef = l_c_coef
        self.xe = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
    
    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        print(batch["attr_scores"])
        return {
            "l_y": self.l_y_coef * self.xe(outputs["class_preds"], batch["class_ids"]),
            "l_c": self.l_c_coef * self.bce(outputs["attr_preds"], batch["attr_scores"])
        }
