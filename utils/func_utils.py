from typing import Union
import torch


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def mvdigamma(vec: torch.FloatTensor, p: int, reduction: str = 'sum'):
    """Batched Multivariate Digamma function
    (https://en.wikipedia.org/wiki/Multivariate_gamma_function#Derivatives)

    Args:
        vec: torch.FloatTensor of shapes (bs, ...), inp to apply function on
        p: int, dimensionality
        reduction: str, one of ['sum', 'mean', 'none'], default 'sum'
    Returns:
        Tensor with same shapes as vec, where the mvdigamma function is
            computed for each position independently
    """
    assert reduction in ['sum', 'mean']

    increasing_numbers = torch.arange(
        1, p + 1, dtype=torch.float, requires_grad=False
    )
    output = torch.digamma(
        vec.unsqueeze(-1) + 0.5 * (1 - increasing_numbers.to(vec.device))
    )

    if reduction == 'sum':
        return output.sum(axis=-1)
    elif reduction == 'mean':
        return output.mean(axis=-1)


def reduce_tensor(vec: torch.Tensor, reduction: str = 'mean'):
    """Global reduction of tensor based on str

    Args:
        vec: torch.FloatTensor
        reduction: str, one of ['sum', 'mean', 'none'], default 'mean'
    """
    assert reduction in ['sum', 'mean', 'none']
    if reduction == 'mean':
        return vec.mean()
    elif reduction == 'sum':
        return vec.sum()
    elif reduction == 'none':
        return vec


def rel_error(value1, value2):
    """Relative difference between two vectors"""
    value1_norm = value1.norm(p=2, dim=-1)
    value2_norm = value2.norm(p=2, dim=-1)
    diff_norm = (value1 - value2).norm(p=2, dim=-1)
    return diff_norm / (value1_norm.pow(0.5) * value2_norm.pow(0.5))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    from scipy.special import digamma

    for k in range(1000):
        rand_nu = torch.randn(512).exp() + 1e-5
        digammas_scipy = torch.zeros(512)
        digammas_torch = mvdigamma(rand_nu, 1)

        for k in range(512):
            digammas_scipy[k] = digamma(rand_nu[k].item())

        if (rel_error(digammas_scipy, digammas_torch) > 1e-6).any():
            raise Exception(
                "Digamma functions of torch and scipy doesn't match"
            )

        if ((rand_nu.log() - 1 / (2 * rand_nu) - digammas_torch) < 0.0).any():
            raise Exception("Upper inequality isn't satisfied")

        if ((rand_nu.log() - 1 / (rand_nu) - digammas_torch) > 0.0).any():
            raise Exception("Lower inequality isn't satisfied")
