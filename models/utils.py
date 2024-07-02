import torch
import torch.nn.functional as F
from torch import nn


class PPNet(nn.Module):
    def __init__(self, backbone, prototype_shape, num_classes, init_weights=True, activation_fn='log'):
        super().__init__()
        self.prototype_shape = prototype_shape
        self.num_prototypes, self.dim, _, _ = prototype_shape
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

        self.features = nn.Sequential(*list(backbone.features.children()))

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=backbone.classifier.in_features, out_channels=self.prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
            nn.Sigmoid()
        )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.register_buffer("ones", torch.ones(self.prototype_shape))
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def _l2_convolution(self, x):
        """
        Compute x ** 2 - 2 * x * prototype + prototype ** 2
        All channels of x2_patch_sum at position i, j have the same values
        All spacial values of p2_reshape at each channel are the same
        """
        x2 = x ** 2  # shape: [b, c, h, w]
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)  # shape: [b, num_prototypes, h, w]

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))  # shape [num_prototypes, ]
        p2_reshape = p2.view(-1, 1, 1) # shape [num_prototypes, 1, 1]

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # p2_reshape broadcasted to [b, num_prototypes, h, wv]
   
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances  # shape: [b, num_prototypes, h, w]

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.features(x)
        conv_features = self.add_on_layers(conv_features)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_to_similarity(self, distances):
        if self.activation_fn == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        else:
            return -distances

    def forward(self, x):
        distances = self.prototype_distances(x)  # shape: [b, num_prototypes, h, w]
        b, num_prototypes, h, w = distances.shape
        # 2D min pooling
        min_distances = -F.max_pool2d(-distances, kernel_size=(h, w,))  # shape: [b, num_prototypes, 1, 1]
        min_distances = min_distances.view(-1, self.num_prototypes)  # shape: [b, num_prototypes]
        prototype_activations = self.distance_to_similarity(min_distances)  # shape: [b, num_prototypes]
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(correct_class_connection * positive_one_weights_locations
                                          + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


class PPNetLoss(nn.Module):
    def __init__(self, l_clst_coef: float, l_sep_coef: float, l_l1_coef: float) -> None:
        super().__init__()
        self.l_clst_coef = l_clst_coef
        self.l_sep_coef = l_sep_coef
        self.l_l1_coef = l_l1_coef
        self.xe = nn.CrossEntropyLoss()

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor],
                batch: dict[str, torch.Tensor],
                prototype_class_identities: torch.Tensor,
                last_layer_weights: torch.Tensor):

        logits, min_distances = outputs
        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, batch["class_ids"])
        if self.l_clst_coef != 0:
            l_clst = self.compute_clst_loss(min_distances, batch["class_ids"], prototype_class_identities)
            loss_dict["l_clst"] = self.l_clst_coef * l_clst
        if self.l_sep_coef != 0:
            l_sep = self.compute_sep_loss(min_distances, batch["class_ids"], prototype_class_identities)
            loss_dict["l_sep"] = self.l_sep_coef * l_sep

        l1_mask = 1 - prototype_class_identities.T
        l1 = (last_layer_weights * l1_mask).norm(p=1)
        loss_dict["l_l1"] = self.l_l1_coef * l1
        return loss_dict

    @staticmethod
    def compute_clst_loss(l2_dists: torch.Tensor,
                          class_ids: torch.Tensor,
                          class_identities: torch.Tensor,
                          dim: int = 1024):
        max_dist = dim
        prototypes_of_correct_class = torch.t(class_identities[:, class_ids])

        inverted_distances, _ = torch.max((max_dist - l2_dists) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)
        return cluster_cost

    @staticmethod
    def compute_sep_loss(l2_dists: torch.Tensor,
                         class_ids: torch.Tensor,
                         class_identities: torch.Tensor,
                         dim: int = 1024):
        # calculate separation cost
        max_dist = dim
        prototypes_of_correct_class = torch.t(class_identities[:, class_ids])

        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_non_target, _ = torch.max((max_dist - l2_dists) * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_non_target)
        return separation_cost
