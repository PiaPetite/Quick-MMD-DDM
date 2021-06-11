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
        X: Tensor with