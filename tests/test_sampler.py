import beanmachine.ppl as bm
import pytest
import torch

from rayleighmachine.sampler import (
    LikelihoodDPP,
    SRAddDeleteInference,
    main
)

__author__ = "Feynman Liang"
__copyright__ = "Feynman Liang"
__license__ = "MIT"


def test_mcmc():
    with pytest.raises(AssertionError):
        LikelihoodDPP(L=torch.randn((3,)))

    with pytest.raises(AssertionError):
        LikelihoodDPP(L=torch.randn((3,2)))

    X = torch.tensor([
        [1, 0, 0],
        [0, 0, 0], # should never be sampled
        [0, 1, 0],
        [0, 0, 1],
    ]).float()
    n, d = X.shape
    L = X @ X.T

    volume_sampling_rows = bm.random_variable(lambda: LikelihoodDPP(L=X @ X.T))

    samples = SRAddDeleteInference().infer(
        queries=[volume_sampling_rows()],
        observations={},
        num_samples=5_000,
        num_chains=1,
        initialize_fn=lambda _: torch.randint(low=0, high=2, size=(n,))
    ).get_chain(0)[volume_sampling_rows()]

    empirical_marginals = samples.float().mean(dim=0)
    assert empirical_marginals[1].item() == 0

    K = L @ torch.inverse(torch.eye(L.shape[0]) + L)
    theoretical_marginals = K.diag()
    assert empirical_marginals.allclose(theoretical_marginals, atol=0.05)

def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main([])
    captured = capsys.readouterr()
    assert "Empirical marginals" in captured.out
