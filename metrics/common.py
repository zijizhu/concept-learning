import torch


@torch.no_grad()
def compute_corrects(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    class_preds = outputs["class_scores"] if isinstance(outputs, dict) else outputs
    class_ids = batch["class_ids"]
    return torch.sum(torch.argmax(class_preds.data, dim=-1) == class_ids.data).item()
