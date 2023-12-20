# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for Metropolis Adjusted Langevin kernels."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss

import geomjax.lmc.diffusions as diffusions
import geomjax.mcmc.proposal as proposal
from geomjax.base import SamplingAlgorithm
from geomjax.types import ArrayLikeTree, ArrayTree, PRNGKey
import geomjax.lmc.metrics as metrics
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cholesky, solve

__all__ = ["MMALAState", "MMMALAState", "init", "build_kernel", "mmala"]


class MALAState(NamedTuple):
    """State of the MALA algorithm.

    The MALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class MMALAState(NamedTuple):
    """Additional information on the MALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_rate: float
    is_accepted: bool


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> MMALAState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)
    return MMALAState(position, logdensity, logdensity_grad)


def build_kernel():
    """Build a MALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def transition_energy(state, new_state, step_size, metric_fn):
        """Transition energy to go from `state` to `new_state`"""
        new_position, _ = ravel_pytree(new_state.position)
        position, _ = ravel_pytree(state.position)
        logdensity_grad, _ = ravel_pytree(state.logdensity_grad)
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)
        d_g = jax.jacfwd(metric_fn)(position)
        if ndim == 1:
            Gamma = jnp.diag(d_g)
            mean_vector = jax.tree_util.tree_map(
                lambda p, g, n: p
                + step_size * (g / metric)
                + Gamma
                + jnp.sqrt(2 * step_size) * n / jnp.sqrt(metric),
                position,
                logdensity_grad,
            )
            covariance_vector = 2 * step_size * metric
            theta_dot = jss.norm(mean_vector, covariance_vector).logpdf(new_position)
        else:
            Gamma = jnp.diag(d_g)
            mean_vector = jax.tree_util.tree_map(
                lambda p, g, n: p
                + step_size * solve(metric, g, assume_a=True)
                + step_size * Gamma,
                position,
                logdensity_grad,
            )
            covariance_matrix = 2 * step_size * metric
            theta_dot = (
                jss.multivariate_normal(mean_vector, covariance_matrix)
                .logpdf(new_position)
                .sum()
            )

        return -state.logdensity + 0.25 * (1.0 / step_size) * theta_dot

    init_proposal, generate_proposal = proposal.asymmetric_proposal_generator(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey,
        state: MMALAState,
        logdensity_fn: Callable,
        step_size: float,
        metric_fn: Callable,
    ) -> tuple[MMALAState, MMALAState]:
        """Generate a new sample with the MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_langevin_riemannian(grad_fn, metric_fn)

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)
        new_state = MMALAState(*new_state)

        proposal = init_proposal(state)
        new_proposal = generate_proposal(state, new_state, step_size=step_size)
        sampled_proposal, do_accept, p_accept = sample_proposal(
            key_rmh, proposal, new_proposal
        )

        info = MMALAState(p_accept, do_accept)

        return sampled_proposal.state, info

    return kernel


class mmala:
    """Implements the (basic) user interface for the MMALA kernel.

    The general mala kernel builder (:meth:`blackjax.mcmc.mala.build_kernel`, alias `blackjax.mala.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new MMALA kernel can be initialized and used with the following code:

    .. code::

        mmala = geomjax.mala(logdensity_fn, step_size)
        state = mmala.init(position)
        new_state, info = mmala.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(mmala.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = blackjax.mala.build_kernel(logdensity_fn)
       state = blackjax.mala.init(position, logdensity_fn, metric_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls, logdensity_fn: Callable, step_size: float, metric_fn: Callable
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn, step_size, metric_fn)

        return SamplingAlgorithm(init_fn, step_fn)
