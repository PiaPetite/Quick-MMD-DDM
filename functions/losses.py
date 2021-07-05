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
    # but changed to not compute the full kernel matrix at once
    m = K_XX.size(0)
    assert K_XX.size() == (m, m)
    assert K_XY.size() == (m, m)
    assert K_YY.size() == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
    else:
        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)

       
    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)
 

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    # Compute the MMD^2 statistic.
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1)) - 2 * K_XY_sum / (m * m)
  
    return mmd2


def _sqn(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten()
    return flat.dot(flat)

def generate_random_projection(n_features: int, n_projections: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates a random projection matrix of shape (n_features, n_projections)
    Args:
        n_features: number of features
        n_projections: number of projections
        device: device to store the matrix
    Returns:
        Tensor of shape (n_features, n_projections)
    """
    return torch.randn((n_features, n_projections), device=device)  
    
class KID(BaseFeatureMetric):
    r"""Interface of Kernel Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. KID can compare two data distributions with different number of samples.
    But dimen