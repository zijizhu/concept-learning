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


class DINOPPNet(nn.Module):
    def __init__(self, backbone: nn.Module, prototype_shape: tuple[int, int], num_classes: int,
                 init_weights=True, activation_fn='log', use_relu=True):
        super().__init__()
        self.prototype_shape = prototype_shape
        self.num_prototypes, self.dim = prototype_shape
        self.num_classes = num_classes
        self.epsilon = 1e-4

        assert activation_fn in ["log", "linear"]
        self.activation_fn = activation_fn

        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.register_buffer("prototype_class_identity", torch.zeros(self.num_prototypes, self.num_classes))

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.backbone = backbone

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        self.register_buffer("ones", torch.ones(self.prototype_shape))
        self.fc = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        self.use_relu = use_relu

        if init_weights:
            self._initialize_weights()
    
    def _l2_distance(self, x):
        '''
        Compute x ** 2 - 2 * x * prototype + prototype ** 2
        All channels of x2_patch_sum at position i, j have the same values
        All spacial values of p2_reshape at each channel are the same
        '''
        x2 = x ** 2  # shape: [b, n_patches, dim]
        x2_patch_sum = x2 @ self.ones.T  # shape: [b, n_patches, n_prototypes]

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=-1)  # shape [n_prototypes, ]
        p2_reshape = p2[None, None]  # shape [1, 1, n_prototypes]

        xp = x @ self.prototype_vectors.T  # shape [b, n_patches, n_prototypes]
        intermediate_result = - 2 * xp + p2_reshape  # p2_reshape broadcasted: [b, n_patches, n_prototypes]

        if self.use_relu:
            distances = F.relu(x2_patch_sum + intermediate_result)

        return distances  # shape: [b, n_patches, n_prototypes]

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        features = self.backbone.forward_features(x)
        patches = features["x_norm_patchtokens"]
        distances = self._l2_distance(patches)
        return distances

    def distance_to_similarity(self, distances):
        """
        Note: Another way to convert could be -x + 1/x
        """
        if self.activation_fn == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        else:
            return -distances

    def forward(self, x):
        distances = self.prototype_distances(x)  # shape: [b, n_patches, n_prototypes]
        distances = distances.permute(0, 2, 1)  # shape: [b, n_prototypes, n_patches]
        b, _, n_patches = distances.shape
        # 2D min pooling
        min_distances = -F.max_pool1d(-distances, kernel_size=n_patches)  # shape: [b, n_prototypes, 1]
        min_distances = min_distances.view(-1, self.num_prototypes)  # shape: [b, num_prototypes]
        prototype_activations = self.distance_to_similarity(min_distances)  # shape: [b, num_prototypes]
        logits = self.fc(prototype_activations)

        return {"attr_preds": logits, "attn_maps": distances, "min_distances": min_distances}

    def _fc_init(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positives = torch.t(self.prototype_class_identity)
        negatives = 1 - positives

        correct_strength = 1
        self.fc.weight.data.copy_(correct_strength * positives  + incorrect_strength * negatives)

    def _initialize_weights(self):
        self._fc_init(incorrect_strength=-0.5)


class DINOPPNetLoss(nn.Module):
    def __init__(self, l_c_coef: float, l_clst_coef: float, l_sep_coef: float, k: int = 5) -> None:
        super().__init__()
        self.l_c_coef = l_c_coef
        self.l_clst_coef = l_clst_coef
        self.l_sep_coef = l_sep_coef
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs:  dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor], prototype_class_identities):
        loss_dict = {}
        loss_dict["l_c"] = self.l_c_coef * self.bce(outputs["attr_preds"], batch["attr_scores"])
        l_clst, l_sep = self.compute_protopnet_losses(outputs["min_distances"],
                                                      batch["attr_scores"],
                                                      prototype_class_identities)
        loss_dict["l_clst"] = self.l_clst_coef * l_clst
        loss_dict["l_sep"] = self.l_sep_coef * l_sep
        return loss_dict
    
    @staticmethod
    def compute_protopnet_losses(l2_dists: torch.Tensor,
                                 binary_attr_labels: torch.Tensor,
                                 class_identities: torch.Tensor):
        """
        Expected shapes:
        l2_dists: [b, n_prototypes,]
        binary_attr_labels: [n, num_attrs]
        class_identities: [n_prototypes, num_attrs]
        """
        cluster_costs, separation_costs = [], []
        batch_active_attr_mask = binary_attr_labels.bool()
        for i, attr_mask in enumerate(batch_active_attr_mask):
            gt_prototype_mask = class_identities[:, attr_mask.bool()]
            gt_prototype_mask = torch.sum(gt_prototype_mask, dim=-1).bool()  # shape: [num_prototypes,]

            min_distances = l2_dists[i, gt_prototype_mask]  # shape: [num_active_attr * k,]
            min_distances = min_distances.view(-1, 5)  # shape: [num_active_attr, k]
            min_distances, _ = torch.min(min_distances, dim=-1) # shape: [num_active_attr,]
            cluster_costs.append(torch.mean(min_distances))

            non_gt_prototype_mask = (1 - gt_prototype_mask.to(torch.long)).bool()
            non_gt_min_distances = l2_dists[i, non_gt_prototype_mask]
            non_gt_min_distances = non_gt_min_distances.view(-1, 5)  # shape: [num_active_attr, k]
            non_gt_min_distances, _ = torch.min(non_gt_min_distances, dim=-1) # shape: [num_active_attr,]
            separation_costs.append(torch.mean(non_gt_min_distances))
        return sum(cluster_costs), sum(separation_costs)


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
