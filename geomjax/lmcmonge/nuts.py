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
"""Public API for the NUTS Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import geomjax.lmcmonge.lmc as lmc
import geomjax.lmcmonge.integrators as integrators
import geomjax.lmcmonge.metrics as metrics
import geomjax.mcmc.proposal as proposal
import geomjax.mcmc.termination as termination
from geomjax.lmcmonge.metrics import lmcmonge_energy
import geomjax.mcmc.trajectory as trajectory
from geomjax.base import SamplingAlgorithm
from geomjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from geomjax.util import hvp
from functools import partial

__all__ = ["NUTSInfo", "init", "build_kernel", "nutslmc"]


init = lmc.init


class NUTSInfo(NamedTuple):
    """Additional information on the NUTS transition.

    This additional information can be used for debugging or computing
    diagnostics.

    velocity:
        The velocity that was sampled and used to integrate the trajectory.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    is_turning
        Whether the sampling returned because the trajectory started turning
        back on itself.
    energy:
        Energy of the transition.
    trajectory_leftmost_state
        The leftmost state of the full trajectory.
    trajectory_rightmost_state
        The rightmost state of the full trajectory.
    num_trajectory_expansions
        Number of subtrajectory samples that were taken.
    num_integration_steps
        Number of integration steps that were taken. This is also the number of
        states in the full trajectory.
    acceptance_rate
        average acceptance probabilty across entire trajectory

    """

    velocity: ArrayTree
    is_divergent: bool
    is_turning: bool
    energy: float
    trajectory_leftmost_state: integrators.RiemannianIntegratorState
    trajectory_rightmost_state: integrators.RiemannianIntegratorState
    num_trajectory_expansions: int
    num_integration_steps: int
    acceptance_rate: float


def build_kernel(
    integrator: Callable = integrators.lan_integrator,
    divergence_threshold: int = 1000,
    max_num_doublings: int = 10,
):
    """Build an iterative NUTS kernel.

    This algorithm is an iteration on the original NUTS algorithm :cite:p:`hoffman2014no`
    with two major differences:

    - We do not use slice samplig but multinomial sampling for the proposal
      :cite:p:`betancourt2017conceptual`;
    - The trajectory expansion is not recursive but iterative :cite:p:`phan2019composable`,
      :cite:p:`lao2020tfp`.

    The implementation can seem unusual for those familiar with similar
    algorithms. Indeed, we do not conceptualize the trajectory construction as
    building a tree. We feel that the tree lingo, inherited from the recursive
    version, is unnecessarily complicated and hides the more general concepts
    upon which the NUTS algorithm is built.

    NUTS, in essence, consists in sampling a trajectory by iteratively choosing
    a direction at random and integrating in this direction a number of times
    that doubles at every step. From this trajectory we continuously sample a
    proposal. When the trajectory turns on itself or when we have reached the
    maximum trajectory length we return the current proposal.

    Parameters
    ----------
    integrator
        The simplectic integrator used to build trajectories.
    divergence_threshold
        The absolute difference in energy above which we consider
        a transition "divergent".
    max_num_doublings
        The maximum number of times we expand the trajectory by
        doubling the number of steps if the trajectory does not
        turn onto itself.

    """

    def kernel(
        rng_key: PRNGKey,
        state: lmc.LMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        stopping_criterion: str = "euc",
    ) -> tuple[lmc.LMCState, NUTSInfo]:
        """Generate a new sample with the NUTS-LMC kernel."""

        (
            alpha2,
            position,
            logdensity,
            logdensity_grad,
            volume_adjustment,
        ) = state

        (
            velocity_generator,
            kinetic_energy_fn,
            set_weighted_gradient,
            normalizing_constant,
            uturn_check_fn,
            metric_vector_product,
        ) = metrics.gaussian_riemannian(alpha2, inverse_mass_matrix)
        uturn_check_fn = partial(uturn_check_fn, criterion=stopping_criterion)
        determinant_metric = normalizing_constant(alpha2, logdensity_grad)
        sqrt_determinant_metric = jnp.sqrt(determinant_metric)
        # normalized quantities
        logdensity_grad_norm = logdensity_grad / sqrt_determinant_metric

        symplectic_integrator = integrator(
            logdensity_fn,
            set_weighted_gradient,
            normalizing_constant,
            metric_vector_product,
        )
        energy_fn = lmcmonge_energy(kinetic_energy_fn)
        proposal_generator = iterative_nuts_proposal(
            symplectic_integrator,
            energy_fn,
            uturn_check_fn,
            max_num_doublings,
            divergence_threshold,
        )

        key_velocity, key_integrator = jax.random.split(rng_key, 2)
        # weighted quantities
        dl_ig = set_weighted_gradient(logdensity_grad_norm)
        Hdl_ig = hvp(logdensity_fn, position, dl_ig) / sqrt_determinant_metric
        ig_Hdl_ig = set_weighted_gradient(Hdl_ig)
        velocity = velocity_generator(key_velocity, position, alpha2, dl_ig)
        momentum = metric_vector_product(
            velocity, alpha2, logdensity_grad, determinant_metric
        )
        # Compute normalized Hvp velocity
        logdensity_hvp_velocity_norm = (
            hvp(logdensity_fn, position, velocity) / sqrt_determinant_metric
        )

        integrator_state = integrators.RiemannianIntegratorState(
            alpha2,
            position,
            momentum,
            velocity,
            logdensity,
            logdensity_grad_norm,
            dl_ig,
            Hdl_ig,
            ig_Hdl_ig,
            logdensity_hvp_velocity_norm,
            determinant_metric,
            volume_adjustment,
        )
        proposal, info = proposal_generator(key_integrator, integrator_state, step_size)
        determinant_metric = proposal.determinant_metric
        sqrt_determinant_metric = jnp.sqrt(proposal.determinant_metric)
        proposal = lmc.LMCState(
            proposal.alpha2,
            proposal.position,
            proposal.logdensity,
            proposal.logdensity_grad_norm * sqrt_determinant_metric,
            proposal.volume_adjustment,
        )
        return proposal, info

    return kernel


class nutslmc:
    """Implements the (basic) user interface for the nuts kernel.

    Examples
    --------

    A new NUTS kernel can be initialized and used with the following code:

    .. code::

        nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
        state = nuts.init(position)
        new_state, info = nuts.step(rng_key, state)

    We can JIT-compile the step function for more speed:

    .. code::

        step = jax.jit(nuts.step)
        new_state, info = step(rng_key, state)

    You can always use the base kernel should you need to:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.nuts.build_kernel(integrators.yoshida)
       state = blackjax.nuts.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the velocity and computing the kinetic energy.
    max_num_doublings
        The maximum number of times we double the length of the trajectory before
        returning if no U-turn has been obserbed or no divergence has occured.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(lmc.init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        *,
        alpha2: float = 0.001,
        max_num_doublings: int = 10,
        divergence_threshold: int = 1000,
        integrator: Callable = integrators.lan_integrator,
        stopping_criterion: str = "euc",
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(integrator, divergence_threshold, max_num_doublings)

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn, alpha2)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                stopping_criterion=stopping_criterion,
            )

        return SamplingAlgorithm(init_fn, step_fn)


def iterative_nuts_proposal(
    integrator: Callable,
    energy_fn: Callable,
    uturn_check_fn: Callable,
    max_num_expansions: int = 10,
    divergence_threshold: float = 1000,
) -> Callable:
    """Iterative NUTS proposal.

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    uturn_check_fn:
        Function that determines whether the trajectory is turning on itself
        (metric-dependant).
    step_size
        Size of the integration step.
    max_num_expansions
        The number of sub-trajectory samples we take to build the trajectory.
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the
    transition.

    """
    (
        new_termination_state,
        update_termination_state,
        is_criterion_met,
    ) = termination.iterative_uturn_numpyro(uturn_check_fn)

    trajectory_integrator = trajectory.dynamic_progressive_integration(
        integrator,
        energy_fn,
        update_termination_state,
        is_criterion_met,
        divergence_threshold,
    )

    expand = trajectory.dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        max_num_expansions,
    )

    def propose(
        rng_key, initial_state: integrators.RiemannianIntegratorState, step_size
    ):
        initial_termination_state = new_termination_state(
            initial_state, max_num_expansions
        )
        initial_energy = energy_fn(initial_state)  # H0 of the HMC step
        initial_proposal = proposal.Proposal(
            initial_state, initial_energy, 0.0, -np.inf
        )
        initial_trajectory = trajectory.Trajectory(
            initial_state,
            initial_state,
            initial_state.momentum,
            initial_state.velocity,
            0,
        )
        initial_expansion_state = trajectory.DynamicExpansionState(
            0, initial_proposal, initial_trajectory, initial_termination_state
        )

        expansion_state, info = expand(
            rng_key, initial_expansion_state, initial_energy, step_size
        )
        is_diverging, is_turning = info
        num_doublings, sampled_proposal, new_trajectory, _ = expansion_state
        # Compute average acceptance probabilty across entire trajectory,
        # even over subtrees that may have been rejected
        acceptance_rate = (
            jnp.exp(sampled_proposal.sum_log_p_accept) / new_trajectory.num_states
        )

        info = NUTSInfo(
            initial_state.velocity,
            is_diverging,
            is_turning,
            sampled_proposal.energy,
            new_trajectory.leftmost_state,
            new_trajectory.rightmost_state,
            num_doublings,
            new_trajectory.num_states,
            acceptance_rate,
        )

        return sampled_proposal.state, info

    return propose
