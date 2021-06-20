import torch
from piq.base import BaseFeatureMetric
from piq.utils import _validate_input


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}



def _polynomial_kernel(X: torch.Tensor, Y: torch.Tensor = None, degree: int = 1, gamma = None,
                       coef0: float = 1.) -> torch.Tensor:
    """
    Compute the polynomial kernel between x and y
    K(X, Y) = (gamma <X, Y> + coef0)^degree
    Args:
        X: Tensor with shape (n_samples_1, n_features)
        Y: torch.Tensor of shape (n_samples_2, n_features)
        degree: default 3
        gamma: if None, defaults to 1.0 / n_features.
        coef0 : default 1
    Returns:
        Gram matrix : Array with shape (n_samples_1, n_samples_2)
    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html
    """

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.size(1)

    K = torch.mm(X, Y.T)
    K *= gamma
    K += coef0
    K.pow_(degree)
    return K


def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor = None, gamma = None) -> torch.Tensor:
    """
    Compute the rbf kernel between X and Y
    K(X, Y) = exp(-1/2sigma ||X - Y||^2)
    Args:
        X: Tensor with shape (n_samples_1, n_features)
        Y: torch.Tensor of shape (n_samples_2, n_features)
        gamma: if None, defaults to 1.0 / n_features.
    Returns:
        Gram matrix : Array with shape (n_samples_1, n_samples_2)
    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html
    """

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.size(1)

    X_norms = (X ** 2).sum(dim=1).view(-1, 1)
    Y_t = Y.t()
    Y_norms = (Y ** 2).sum(dim=1).view(1, -1)

    K = X_norms + Y_norms - 2.0 * torch.mm(X, Y_t)
    K *= -gamma
    K.exp_()
    return K




def _mmd2_and_variance(K_XX: torch.Tensor, K_XY: torch.Tensor, K_YY: torch.Tensor, unit_diagonal: bool = False,
                       mmd_est: str = 'unbiased', var_at_m = None, ret_var: bool = False): 
        
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
  