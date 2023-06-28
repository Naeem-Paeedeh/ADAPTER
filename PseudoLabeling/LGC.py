# This code is modified from https://github.com/provezano/lgc
import torch
from utils.other_utilities import pinv_robust


class LGC:
    """
    Learning with Local and Global Consistency (LGC) algorithm
    """
    def __init__(self, alpha=0.99, sigma=50, rcond_for_pinv=1e-1):
        self.device = None
        self.alpha = alpha
        self.sigma = sigma
        self.rcond_for_pinv = rcond_for_pinv
        self.identity = None
        self.device = None

    def _calculate_A_hat(self, A):
        diag_vec = A.sum(dim=1)
        D_sqrt_inv = torch.diag(1.0 / diag_vec.sqrt())
        return D_sqrt_inv @ A @ D_sqrt_inv

    def compute(self, x, y_bar, logger=None):
        # Regarding the labeling, the first part of the Y_bar must be the labeled part of the target domain.
        # The rest is dedicated to the unlabeled samples. Therefore, the first part of Y_bar is one-hot encoded,
        # but the rest elements of the remaining parts for unlabeled samples must be zeros.

        if self.device is None:
            self.device = 'cpu'     # x.device

        x = x.cpu()   # .to(self.device)
        y_bar = y_bar.to(self.device)
        if self.identity is not None:
            self.identity = self.identity   # .to(self.device)

        # "cdist computes batched the p-norm distance between each pair of the two collections of row vectors"
        distance_matrix = torch.cdist(x, x)
        # A is the affinity matrix
        A = torch.exp(-(distance_matrix * distance_matrix) / (2.0 * (self.sigma ** 2)))   # RBF
        A.fill_diagonal_(0)
        A_hat = self._calculate_A_hat(A)
        if self.identity is None:
            self.identity = torch.eye(y_bar.shape[0])   # .to(self.device)
        # f_star = (self.identity - self.alpha * A_hat).cpu().pinverse(self.rcond_for_pinv).to(self.device) @ y_bar
        f_star = pinv_robust(self.identity - self.alpha * A_hat, self.rcond_for_pinv, self.device) @ y_bar

        return f_star.to(self.device)
