import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import pairwise_cosine_similarity


def _mean_squared_mahalanobis(x: torch.Tensor, mu: torch.Tensor, sigma_inv):
    '''Computes the mean of squared mahalanobis distances from a vector or a set of vectors to the distribution
    with mean my and and inverse covariant matrix sigma_inv.
    Implementation from https://github.com/wangyu-ustc/LM4CV/blob/main/utils/train_utils.py#L263
    
    Args:
        x (Tensor[M, D]) or (Tensor[D]): a vector or a set of vector of length D.
        distribution (Tensor[N, D]) a matrix of N vectors of length D
    
    Returns:
        Tensor[]: a scaler tensor, which is the mahalanobis distance from vec to the distribution.
    '''
    delta = x - mu.unsqueeze(0)
    return torch.diag(delta @ sigma_inv @ delta.T).mean()


class MahalanobisCriterion(nn.Module):
    '''Cross-entropy loss with mahalanobis distance regularization
    '''
    def __init__(self, regularization=True, division_power=3) -> None:
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.regularization = regularization
        self.division_power = division_power
    
    def forward(self, outputs: torch.Tensor, targets, weights, target_dis):
        xe_loss = self.xe(outputs, targets)
        if not self.regularization:
            return xe_loss

        # Original implementation from https://github.com/wangyu-ustc/LM4CV/blob/main/utils/train_utils.py#L208
        # which is different to the one described in the paper.
        weights_norm = torch.linalg.vector_norm(weights, dim=-1, keepdim=True)
        mu = torch.mean(target_dis, dim=0)
        sigma_inv = torch.inverse(torch.cov(target_dis.T)).to(outputs.device)    # Using torch.inverse will have different result to np.linalg.inv
        # Alternate implementation: sigma_inv = torch.inverse(torch.cov(distribution.T))

        mean_distance = torch.stack([_mean_squared_mahalanobis(concept, mu, sigma_inv)
                                     for concept
                                     in target_dis]).mean().to(outputs.device)

        mahalanobis_loss = _mean_squared_mahalanobis(weights / weights_norm, mu, sigma_inv)
        mahalanobis_loss_scaled = (mahalanobis_loss - mean_distance) / (mean_distance ** self.division_power)

        return xe_loss + torch.abs(mahalanobis_loss_scaled), xe_loss, mahalanobis_loss_scaled


def mmd(x, y, sigma):
    # Implementation from https://torchdrift.org/notebooks/note_on_mmd.html
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1/(2*sigma**2)) * dists**2) + torch.eye(n+m)*1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd

class MMDCriterion(nn.Module):
    def __init__(self, mmd_coef: float=100) -> None:
        super(MMDCriterion, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mmd_coef = mmd_coef
    
    def forward(
            self,
            preds: torch.Tensor,
            tgts: torch.Tensor,
            weights: torch.Tensor,
            weights_tgt: torch.Tensor):
        '''Assume weights_tgt is already normalized'''
        assert weights.shape[-1] == weights_tgt.shape[-1]
        weights /= weights.norm(p=2, dim=-1, keepdim=True)
        dists = torch.pdist(torch.cat([weights, weights_tgt], dim=0))
        sigma = dists.median()/2
        return self.cross_entropy(preds, tgts) + self.mmd_coef * mmd(weights, weights_tgt, sigma)


class ConceptRetrievalModel(nn.Module):
    def __init__(self, concepts_encoded, k, num_classes, retrieval_algo: str = 'hungarian') -> None:
        super(ConceptRetrievalModel, self).__init__()
        self.k = k
        self.dim = concepts_encoded.shape[-1]
        self.register_buffer('all_concepts', concepts_encoded)
        self.prototypes = nn.Parameter(torch.randn(k, self.dim))
        self.retrieved_concepts = None
        self.classifier = nn.Linear(k, num_classes)
        assert retrieval_algo in ['greedy', 'hungarian']
        self.retrieval_algo = retrieval_algo
    
    def forward(self, x):
        if self.retrieved_concepts is not None:
            concept_emb = self.retrieved_concepts
        else:
            concept_emb = self.prototypes
        concept_logits = x @ concept_emb.T
        class_logits = self.classifier(concept_logits)
        return concept_logits, class_logits

    @torch.no_grad()
    def match_concepts(self):
        if self.retrieval_algo == 'greedy':
            selected_idxs = []
            for cp in tqdm(self.prototypes):
                cp = cp/ torch.linalg.vector_norm(cp)
                similarities = F.cosine_similarity(cp, self.all_concepts)
                sorted_idxs = torch.argsort(similarities)
                count = 0
                while sorted_idxs[count] in selected_idxs:
                    count += 1
                selected_idxs.append(sorted_idxs[count])
            selected_idxs = np.array(selected_idxs)
            self.register_buffer('retrieved_concepts', self.all_concepts[selected_idxs])
            return selected_idxs
        elif self.retrieval_algo == 'hungarian':
            cost_matrix = pairwise_cosine_similarity(self.prototypes, self.all_concepts)
            prototype_idxs, concept_idxs = linear_sum_assignment(cost_matrix)
            self.register_buffer('retrieved_concepts', self.all_concepts[concept_idxs])
            return concept_idxs
        else:
            raise NotImplementedError
