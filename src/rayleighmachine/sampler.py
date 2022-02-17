"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = rayleighmachine.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This skeleton file can be safely removed if not needed!

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.base_single_site_proposer import \
    BaseSingleSiteMHProposer
from beanmachine.ppl.inference.single_site_inference import SingleSiteInference
from beanmachine.ppl.world import World

from rayleighmachine import __version__

__author__ = "Feynman Liang"
__copyright__ = "Feynman Liang"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from rayleighmachine.skeleton import fib`,
# when using this Python module as a library.


class AddDeleteProposal(dist.Distribution):
    def __init__(self, state: torch.tensor):
        super().__init__()
        self.state = state

    def sample(self):
        "https://arxiv.org/pdf/1607.03559.pdf"
        state = self.state.clone()

        b = torch.rand(size=(1,)).item() > 0.5
        if b:
            t = torch.randint(low=0, high=state.shape[0], size=(1,)).item()
            if state[t].item() == 1:
                # delete
                state[t] = 0
                return state
            else:
                # add
                state[t] = 1
                return state
        else:
            # do nothing
            return state

    def log_prob(self, value):
        # constant log_prob to exclude from MH ratio
        return torch.tensor(0.0)


class SRAddDeleteProposer(BaseSingleSiteMHProposer):
    "Assumes the only node ever sampled is the SR set-valued RV"

    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        return AddDeleteProposal(world.get_variable(self.node).value)


class SRAddDeleteInference(SingleSiteInference):
    """Add/Delete MH MCMC sampler for a SR set-valued measure"""

    def __init__(self) -> None:
        super().__init__(SRAddDeleteProposer)


class LikelihoodDPP(dist.Distribution):
    "DPP with likelihood kernel L"

    def __init__(self, L: torch.tensor) -> None:
        assert (
            L.dim() == 2 and L.shape[0] == L.shape[1]
        ), "L must be a square PSD matrix"
        super().__init__()

        self.L = L
        n = L.shape[0]
        self.Z = (torch.eye(n) + L).det()

    def log_prob(self, state: torch.tensor):
        S = state == 1
        L_S = self.L[S][:, S]
        return torch.log(L_S.det() / self.Z)


def sample_mcmc(L, num_samples, num_chains, **inference_args):
    d = L.shape[0]

    volume_sampling_rows = bm.random_variable(lambda: LikelihoodDPP(L=L))

    samples = (
        SRAddDeleteInference()
        .infer(
            queries=[volume_sampling_rows()],
            observations={},
            initialize_fn=lambda _: torch.randint(low=0, high=2, size=(d,)),
            num_samples=num_samples,
            num_chains=num_chains,
            **inference_args,
        )
        .get_chain(0)[volume_sampling_rows()]
    )

    print(f"Empirical marginals: {samples.float().mean(dim=0)}")

    K = L @ torch.inverse(torch.eye(L.shape[0]) + L)
    print(f"Theoretical marginals: {K.diag()}")


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="A DPP sampler using beanmachine")
    parser.add_argument(
        "--version",
        action="version",
        version="RayleighMachine {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`sample_mcmc` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`sample_mcmc`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")

    X = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, 0],  # should never be sampled
            [0, 1, 0],
            [0, 0, 1],
        ]
    ).float()
    L = X @ X.T
    sample_mcmc(L=L, num_samples=1_000, num_chains=1)
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m rayleighmachine.skeleton 42
    #
    run()
