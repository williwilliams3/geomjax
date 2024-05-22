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

from typing import Callable, NamedTuple, Union

import jax

import geomjax.mcmc.proposal as proposal
import geomjax.rmhmc.integrators as integrators
import geomjax.rmhmc.metrics as metrics
import geomjax.mcmc.trajectory as trajectory
from geomjax.base import SamplingAlgorithm
from geomjax.mcmc.metrics import hmc_energy
from geomjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

__all__ = ["init", "build_kernel", "rmhmc"]


class RMHMCState(NamedTuple):
    """State of the RMHMCState algorithm.

    The RMHMCState algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current logdensity as well as the current gradient of the logdensity.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class DynamicRMHMCState(NamedTuple):
    """State of the dynamic RMHMC algorithm.

    Adds a utility array for generating a pseudo or quasi-random sequence of
    number of integration steps.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    random_generator_arg: Array


class RMHMCInfo(NamedTuple):
    """Additional information on the RMHMCSTATE transition.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum:
        The momentum that was sampled and used to integrate the trajectory.
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
        momentum.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory

    """

    momentum: ArrayTree
    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: integrators.IntegratorState
    num_integration_steps: int


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return RMHMCState(position, logdensity, logdensity_grad)


def init_dynamic(
    position: ArrayLikeTree, logdensity_fn: Callable, random_generator_arg: Array
):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return DynamicRMHMCState(
        position, logdensity, logdensity_grad, random_generator_arg
    )


def build_kernel(
    integrator: Callable = integrators.implicit_midpoint,
    divergence_threshold: float = 1000,
):
    """Build a RMHMC kernel.

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
        state: RMHMCState,
        logdensity_fn: Callable,
        step_size: float,
        metric_fn: Callable,
        num_integration_steps: int,
    ) -> tuple[RMHMCState, RMHMCInfo]:
        """Generate a new sample with the RMHMC kernel."""

        (
            momentum_generator,
            kinetic_energy_fn,
            _,
            inverse_metric_vector_product,
        ) = metrics.gaussian_riemannian(metric_fn)
        symplectic_integrator = integrator(
            logdensity_fn, kinetic_energy_fn, inverse_metric_vector_product
        )
        proposal_generator = rmhmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = momentum_generator(key_momentum, position)
        velocity = inverse_metric_vector_product(position, momentum)

        integrator_state = integrators.IntegratorState(
            position, momentum, velocity, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = RMHMCState(
            proposal.position,
            proposal.logdensity,
            proposal.logdensity_grad,
        )

        return proposal, info

    return kernel


def build_dynamic_kernel(
    integrator: Callable = integrators.implicit_midpoint,
    divergence_threshold: float = 1000,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
):
    """Build a Dynamic RMHMC kernel where the number of integration steps is chosen randomly.

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
        state: DynamicRMHMCState,
        logdensity_fn: Callable,
        step_size: float,
        metric_fn: Array,
        **integration_steps_kwargs,
    ) -> tuple[DynamicRMHMCState, RMHMCInfo]:
        """Generate a new sample with the LMC kernel."""
        num_integration_steps = integration_steps_fn(
            state.random_generator_arg, **integration_steps_kwargs
        )
        lmc_state = RMHMCState(
            state.position,
            state.logdensity,
            state.logdensity_grad,
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
            DynamicRMHMCState(
                lmc_proposal.position,
                lmc_proposal.logdensity,
                lmc_proposal.logdensity_grad,
                next_random_arg,
            ),
            info,
        )

    return kernel


class rmhmc:
    """A Riemannian Manifold Hamiltonian Monte Carlo kernel

    Of note, this kernel is simply an alias of the ``hmc`` kernel with a
    different choice of default integrator (``implicit_midpoint`` instead of
    ``momentum_verlet``) since RMHMC is typically used for Hamiltonian systems
    that are not separable.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    mass_matrix
        A function which computes the mass matrix (not inverse) at a given
        position when drawing a value for the momentum and computing the kinetic
        energy. In practice, this argument will be passed to the
        ``metrics.default_metric`` function so it supports all the options
        discussed there.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the
        trajectory.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
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
        integrator: Callable = integrators.implicit_midpoint,
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


class dynamic_rmhmc:
    """Implements the (basic) user interface for the dynamic RMHMC kernel.

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
        integrator: Callable = integrators.implicit_midpoint,
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


def rmhmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: Union[float, ArrayLikeTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    """Vanilla RMHMC algorithm.

    The algorithm integrates the trajectory applying a symplectic integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. This is what people usually refer to when they talk about "the
    RMHMC algorithm".

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
        hmc_energy(kinetic_energy)
    )

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> tuple[integrators.IntegratorState, RMHMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal = generate_proposal(proposal.energy, end_state)
        is_diverging = -new_proposal.weight > divergence_threshold
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = RMHMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate


def flip_momentum(
    state: integrators.IntegratorState,
) -> integrators.IntegratorState:
    """Flip the momentum at the end of the trajectory.

    To guarantee time-reversibility (hence detailed balance) we
    need to flip the last state's momentum. If we run the hamiltonian
    dynamics starting from the last state with flipped momentum we
    should indeed retrieve the initial state (with flipped momentum).

    """
    flipped_momentum = jax.tree_util.tree_map(lambda m: -1.0 * m, state.momentum)
    flipped_velocity = jax.tree_util.tree_map(lambda m: -1.0 * m, state.velocity)
    return integrators.IntegratorState(
        state.position,
        flipped_momentum,
        flipped_velocity,
        state.logdensity,
        state.logdensity_grad,
    )
