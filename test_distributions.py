from distributions.diagonal_normal_wishart import test_diagonal_normal_wishart
from distributions.mixture_gaussian import test_gaussian_mixture
from distributions.nw_prior import test_nw_prior

if __name__ == '__main__':
    test_diagonal_normal_wishart()
    test_gaussian_mixture()
    test_nw_prior()
    print("All test passed")
