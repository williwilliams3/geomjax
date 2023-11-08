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
"""Public API for the Generalized (Non-reversible w/ persistent velocity) LMC Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import geomjax.rmcmc.lmc as lmc
import geomjax.mcmc.integrators as integrators
import geomjax.rmcmc.metrics as metrics
import geomjax.mcmc.proposal as proposal
from geomjax.base import SamplingAlgorithm
from geomjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from geomjax.util import generate_gaussian_noise, pytree_size

__all__ = ["GLMCState", "init", "build_kernel", "glmc"]


class GLMCState(NamedTuple):
    """State of the Generalized LMC algorithm.

    The Generalized LMC algorithm is persistent on its velocity, hence
    taking as input a position and velocity pair, updating and returning
    it for the next iteration. The algorithm also uses a persistent slice
    to perform a non-reversible Metropolis Hastings update, thus we also
    store the current slice variable and return its updated version after
    each iteration. To make computations more efficient, we also store
    the current logdensity as well as the current gradient of the
    logdensity.

    """

    position: ArrayTree
    velocity: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    slice: float
    volume_adjustment: float


def init(
    position: ArrayLikeTree,
    rng_key: PRNGKey,
    logdensity_fn: Callable,
) -> GLMCState:
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)

    key_velocity, key_slice = jax.random.split(rng_key)
    velocity = generate_gaussian_noise(key_velocity, position)
    slice = jax.random.uniform(key_slice, minval=-1.0, maxval=1.0)
    volume_adjustment = 0.0

    return GLMCState(
        position, velocity, logdensity, logdensity_grad, slice, volume_adjustment
    )


def build_kernel(
    noise_fn: Callable = lambda _: 0.0,
    divergence_threshold: float = 1000,
):
    """Build a Generalized LMC kernel.

    The Generalized LMC kernel performs a similar procedure to the standard LMC
    kernel with the difference of a persistent velocity variable and a non-reversible
    Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
    step. This means that; apart from velocity and slice variables that are dependent
    on the previous velocity and slice variables, and a Metropolis-Hastings step
    performed (equivalently) as slice sampling; the standard LMC's implementation can
    be re-used to perform Generalized LMC sampling.

    Parameters
    ----------
    noise_fn
        A function that takes as input the slice variable and outputs a random
        variable used as a noise correction of the persistent slice update.
        The parameter defaults to a random variable with a single atom at 0.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key, a Pytree that contains the current state
    of the chain, and free parameters of the sampling mechanism; and that
    returns a new state of the chain along with information about the transition.

    """
    sample_proposal = proposal.nonreversible_slice_sampling

    def kernel(
        rng_key: PRNGKey,
        state: GLMCState,
        logdensity_fn: Callable,
        step_size: float,
        velocity_inverse_scale: ArrayLikeTree,
        alpha: float,
        delta: float,
    ) -> tuple[GLMCState, lmc.LMCInfo]:
        """Generate new sample with the Generalized LMC kernel.

        Parameters
        ----------
        rng_key
            JAX's pseudo random number generating key.
        state
            Current state of the chain.
        logdensity_fn
            (Unnormalized) Log density function being targeted.
        step_size
            Variable specifying the size of the integration step.
        velocity_inverse_scale
            Pytree with the same structure as the targeted position variable
            specifying the per dimension inverse scaling transformation applied
            to the persistent velocity variable prior to the integration step.
        alpha
            Variable specifying the degree of persistent velocity, complementary
            to independent new velocity.
        delta
            Fixed (non-random) amount of translation added at each new iteration
            to the slice variable for non-reversible slice sampling.

        """

        flat_inverse_scale = jax.flatten_util.ravel_pytree(velocity_inverse_scale)[0]
        _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(flat_inverse_scale**2)

        symplectic_integrator = integrators.velocity_verlet(
            logdensity_fn, kinetic_energy_fn
        )
        proposal_generator = lmc.lmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            divergence_threshold=divergence_threshold,
            sample_proposal=sample_proposal,
        )

        key_velocity, key_noise = jax.random.split(rng_key)
        position, velocity, logdensity, logdensity_grad, slice = state
        # New velocity is persistent
        velocity = update_velocity(key_velocity, state, alpha)
        velocity = jax.tree_map(lambda m, s: m / s, velocity, velocity_inverse_scale)
        # Slice is non-reversible
        slice = ((slice + 1.0 + delta + noise_fn(key_noise)) % 2) - 1.0

        integrator_state = integrators.IntegratorState(
            position, velocity, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(slice, integrator_state)
        proposal = lmc.flip_velocity(proposal)
        state = GLMCState(
            proposal.position,
            jax.tree_map(lambda m, s: m * s, proposal.velocity, velocity_inverse_scale),
            proposal.logdensity,
            proposal.logdensity_grad,
            info.acceptance_rate,
        )

        return state, info

    return kernel


def update_velocity(rng_key, state, alpha):
    """Persistent update of the velocity variable.

    Performs a persistent update of the velocity, taking as input the previous
    velocity, a random number generating key and the parameter alpha. Outputs
    an updated velocity that is a mixture of the previous velocity a new sample
    from a Gaussian density (dependent on alpha). The weights of the mixture of
    these two components are a function of alpha.

    """
    position, velocity, *_ = state

    m_size = pytree_size(velocity)
    velocity_generator, *_ = metrics.gaussian_euclidean(1 / alpha * jnp.ones((m_size,)))
    velocity = jax.tree_map(
        lambda prev_velocity, shifted_velocity: prev_velocity * jnp.sqrt(1.0 - alpha)
        + shifted_velocity,
        velocity,
        velocity_generator(rng_key, position),
    )

    return velocity


class glmc:
    """Implements the (basic) user interface for the Generalized LMC kernel.

    The Generalized LMC kernel performs a similar procedure to the standard LMC
    kernel with the difference of a persistent velocity variable and a non-reversible
    Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
    step.

    This means that the sampling of the velocity variable depends on the previous
    velocity, the rate of persistence depends on the alpha parameter, and that the
    Metropolis-Hastings accept/reject step is done through slice sampling with a
    non-reversible slice variable also dependent on the previous slice, the determinisitc
    transformation is defined by the delta parameter.

    The Generalized LMC does not have a trajectory length parameter, it always performs
    one iteration of the velocity verlet integrator with a given step size, making
    the algorithm a good candiate for running many chains in parallel.

    Examples
    --------

    A new Generalized LMC kernel can be initialized and used with the following code:

    .. code::

        glmc_kernel = blackjax.glmc(logdensity_fn, step_size, alpha, delta)
        state = glmc_kernel.init(rng_key, position)
        new_state, info = glmc_kernel.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(glmc_kernel.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        A PyTree of the same structure as the target PyTree (position) with the
        values used for as a step size for each dimension of the target space in
        the velocity verlet integrator.
    alpha
        The value defining the persistence of the velocity variable.
    delta
        The value defining the deterministic translation of the slice variable.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    noise_gn
        A function that takes as input the slice variable and outputs a random
        variable used as a noise correction of the persistent slice update.
        The parameter defaults to a random variable with a single atom at 0.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        velocity_inverse_scale: ArrayLikeTree,
        alpha: float,
        delta: float,
        *,
        divergence_threshold: int = 1000,
        noise_gn: Callable = lambda _: 0.0,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(noise_gn, divergence_threshold)

        def init_fn(position: ArrayLikeTree, rng_key: PRNGKey):
            return cls.init(position, rng_key, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                velocity_inverse_scale,
                alpha,
                delta,
            )

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
