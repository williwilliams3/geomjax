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
"""Public API for the LMC Kernel"""
from typing import Callable, NamedTuple, Union

import jax

import geomjax.mcmc.proposal as proposal
import geomjax.lmcmc.integrators as integrators
import geomjax.lmcmc.metrics as metrics
import geomjax.lmcmc.trajectory as trajectory
from geomjax.base import SamplingAlgorithm
from geomjax.lmcmc.trajectory import lmc_energy
from geomjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

__all__ = ["LMCState", "LMCInfo", "init", "build_kernel", "lmc"]


class LMCState(NamedTuple):
    """State of the LMC algorithm.

    The LMC algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current logdensity as well as the current gradient of the logdensity.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    volume_adjustment: float


class DynamicLMCState(NamedTuple):
    """State of the dynamic LMC algorithm.

    Adds a utility array for generating a pseudo or quasi-random sequence of
    number of integration steps.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    volume_adjustment: float
    random_generator_arg: Array


class LMCInfo(NamedTuple):
    """Additional information on the LMC transition.

    This additional information can be used for debugging or computing
    diagnostics.

    velocity:
        The velocity that was sampled and used to integrate the trajectory.
    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    energy:
        Total energy of the transition.
    proposal
        The state proposed by the proposal. Typically includes the position and
        velocity.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory

    """

    velocity: ArrayTree
    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: integrators.IntegratorState
    num_integration_steps: int


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    volume_adjustment = 0.0
    return LMCState(position, logdensity, logdensity_grad, volume_adjustment)


def init_dynamic(
    position: ArrayLikeTree, logdensity_fn: Callable, random_generator_arg: Array
):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    volume_adjustment = 0.0
    return DynamicLMCState(
        position, logdensity, logdensity_grad, volume_adjustment, random_generator_arg
    )


def build_kernel(
    integrator: Callable = integrators.lan_integrator,
    divergence_threshold: float = 1000,
):
    """Build a LMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: LMCState,
        logdensity_fn: Callable,
        step_size: float,
        metric_fn: Callable,
        num_integration_steps: int,
    ) -> tuple[LMCState, LMCInfo]:
        """Generate a new sample with the LMC kernel."""

        (
            velocity_generator,
            kinetic_energy_fn,
            _,
            omega_tilde_fn,
            grad_logdetmetric,
            metric_vector_product,
        ) = metrics.gaussian_riemannian(metric_fn)
        symplectic_integrator = integrator(
            logdensity_fn, omega_tilde_fn, grad_logdetmetric, metric_vector_product
        )
        proposal_generator = lmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_velocity, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad, volume_adjustment = state
        velocity = velocity_generator(key_velocity, position)

        integrator_state = integrators.IntegratorState(
            position, velocity, logdensity, logdensity_grad, volume_adjustment
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = LMCState(
            proposal.position,
            proposal.logdensity,
            proposal.logdensity_grad,
            proposal.volume_adjustment,
        )

        return proposal, info

    return kernel


def build_dynamic_kernel(
    integrator: Callable = integrators.lan_integrator,
    divergence_threshold: float = 1000,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
):
    """Build a Dynamic LMC kernel where the number of integration steps is chosen randomly.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    integration_steps_fn
        Function that generates the next pseudo or quasi-random number of integration steps in the
        sequence, given the current `random_generator_arg`. Needs to return an `int`.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    lmc_base = build_kernel(integrator, divergence_threshold)

    def kernel(
        rng_key: PRNGKey,
        state: DynamicLMCState,
        logdensity_fn: Callable,
        step_size: float,
        metric_fn: Array,
        **integration_steps_kwargs,
    ) -> tuple[DynamicLMCState, LMCInfo]:
        """Generate a new sample with the LMC kernel."""
        num_integration_steps = integration_steps_fn(
            state.random_generator_arg, **integration_steps_kwargs
        )
        lmc_state = LMCState(
            state.position,
            state.logdensity,
            state.logdensity_grad,
            state.volume_adjustment,
        )
        lmc_proposal, info = lmc_base(
            rng_key,
            lmc_state,
            logdensity_fn,
            step_size,
            metric_fn,
            num_integration_steps,
        )
        next_random_arg = next_random_arg_fn(state.random_generator_arg)
        return (
            DynamicLMCState(
                lmc_proposal.position,
                lmc_proposal.logdensity,
                lmc_proposal.logdensity_grad,
                lmc_proposal.volume_adjustment,
                next_random_arg,
            ),
            info,
        )

    return kernel


class lmc:
    """Implements the (basic) user interface for the LMC kernel.

    The general lmc kernel builder (:meth:`blackjax.mcmc.lmc.build_kernel`, alias `blackjax.lmc.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.lmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new LMC kernel can be initialized and used with the following code:

    .. code::

        lmc = blackjax.lmc(logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
        state = lmc.init(position)
        new_state, info = lmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(lmc.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.lmc.build_kernel(integrators.mclachlan)
       state = blackjax.lmc.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    metric_fn
        The value to use for the metric_fn(position)  when drawing a value for
        the velocity and computing the kinetic energy.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.\

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
        metric_fn: Callable,
        num_integration_steps: int,
        *,
        divergence_threshold: int = 1000,
        integrator: Callable = integrators.lan_integrator,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(integrator, divergence_threshold)

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                metric_fn,
                num_integration_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)


class dynamic_lmc:
    """Implements the (basic) user interface for the dynamic LMC kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    metric_fn
        The metric function.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    integration_steps_fn
        Function that generates the next pseudo or quasi-random number of integration steps in the
        sequence, given the current `random_generator_arg`.


    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    init = staticmethod(init_dynamic)
    build_kernel = staticmethod(build_dynamic_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        metric_fn: Callable,
        *,
        divergence_threshold: int = 1000,
        integrator: Callable = integrators.lan_integrator,
        next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
        integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(
            integrator, divergence_threshold, next_random_arg_fn, integration_steps_fn
        )

        def init_fn(position: ArrayLikeTree, random_generator_arg: Array):
            return cls.init(position, logdensity_fn, random_generator_arg)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                metric_fn,
            )

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


def lmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: Union[float, ArrayLikeTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    """Vanilla LMC algorithm.

    The algorithm integrates the trajectory applying a symplectic integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. This is what people usually refer to when they talk about "the
    LMC algorithm".

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

    """
    build_trajectory = trajectory.static_integration(integrator)
    init_proposal, generate_proposal = proposal.proposal_generator(
        lmc_energy(kinetic_energy)
    )

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> tuple[integrators.IntegratorState, LMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = flip_velocity(end_state)
        proposal = init_proposal(state)
        new_proposal = generate_proposal(proposal.energy, end_state)
        is_diverging = -new_proposal.weight > divergence_threshold
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = LMCInfo(
            state.velocity,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate


def flip_velocity(
    state: integrators.IntegratorState,
) -> integrators.IntegratorState:
    """Flip the velocity at the end of the trajectory.

    To guarantee time-reversibility (hence detailed balance) we
    need to flip the last state's velocity. If we run the hamiltonian
    dynamics starting from the last state with flipped velocity we
    should indeed retrieve the initial state (with flipped velocity).

    """
    flipped_velocity = jax.tree_util.tree_map(lambda m: -1.0 * m, state.velocity)
    return integrators.IntegratorState(
        state.position,
        flipped_velocity,
        state.logdensity,
        state.logdensity_grad,
        state.volume_adjustment,
    )
