import timm
import torch
import torch.nn.functional as F
from torch import nn, optim


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
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.backbone = backbone

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.register_buffer("ones", torch.ones(self.prototype_shape))
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def _l2_convolution(self, x):
        '''
        Compute x ** 2 - 2 * x * prototype + prototype ** 2
        All channels of x2_patch_sum at position i, j have the same values
        All spacial values of p2_reshape at each channel are the same
        '''
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
        conv_features = self.backbone(x)
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
    #     for m in self.add_on_layers.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # every init technique has an underscore _ in the name
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


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

    loss = F.cross_entropy(preds, batch['class_ids'])
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
