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
"""Implementation of the Stan warmup for the HMC family of sampling algorithms."""
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

import geomjax.lmcmc as lmcmc
from geomjax.adaptation.base import AdaptationInfo, AdaptationResults

from geomjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from geomjax.base import AdaptationAlgorithm
from geomjax.progress_bar import progress_bar_scan
from geomjax.types import ArrayLikeTree, PRNGKey

__all__ = ["WindowAdaptationState", "base", "build_schedule", "window_adaptation"]


class StepSizeAdaptationState(NamedTuple):
    ss_state: DualAveragingAdaptationState  # step size
    step_size: float


def base(
    target_acceptance_rate: float = 0.80,
) -> tuple[Callable, Callable, Callable]:
    """
    Parameters
    ----------
    target_acceptance_rate:
        The target acceptance rate for the step size adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup
        state.

    """
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        position: ArrayLikeTree, initial_step_size: float
    ) -> StepSizeAdaptationState:
        """Initialze the adaptation state and parameter values."""

        ss_state = da_init(initial_step_size)

        return StepSizeAdaptationState(
            ss_state,
            initial_step_size,
        )

    def update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: StepSizeAdaptationState,
    ) -> StepSizeAdaptationState:
        """Update the adaptation state when in a "fast" window.

        Only the step size is adapted in fast windows. "Fast" refers to the fact
        that the optimization algorithms are relatively fast to converge
        compared to the covariance estimation with Welford's algorithm

        """
        del position

        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return StepSizeAdaptationState(new_ss_state, new_step_size)

    def final(warmup_state: StepSizeAdaptationState) -> float:
        """Return the final values for the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        return step_size

    return init, update, final


def step_size_adaptation(
    algorithm: Union[lmcmc.lmc.lmc, lmcmc.nuts.nuts],
    logdensity_fn: Callable,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    lower_bound: float = 1e-3,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the step size parameters of
    algorithms in the HMC fmaily.


    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to
        sample.

    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    progress_bar
        Whether we should display a progress bar.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that runs the adaptation and returns an `AdaptationResult` object.

    """

    mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_step, adapt_final = base(
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, xs):
        _, rng_key = xs
        state, adaptation_state = carry

        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size=adaptation_state.step_size,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            new_state.position,
            info.acceptance_rate,
            adaptation_state,
        )

        return (
            (new_state, new_adaptation_state),
            AdaptationInfo(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        if algorithm == lmcmc.mmala.mmala:
            init_state = algorithm.init(
                position, logdensity_fn, extra_parameters["metric_fn"]
            )
        else:
            init_state = algorithm.init(position, logdensity_fn)
        init_adaptation_state = adapt_init(position, initial_step_size)

        if progress_bar:
            print("Running window adaptation")
            one_step_ = jax.jit(progress_bar_scan(num_steps)(one_step))
        else:
            one_step_ = jax.jit(one_step)

        keys = jax.random.split(rng_key, num_steps)

        last_state, info = jax.lax.scan(
            one_step_,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys),
        )
        last_chain_state, last_warmup_state, *_ = last_state

        step_size = adapt_final(last_warmup_state)
        parameters = {
            "step_size": jnp.maximum(step_size, lower_bound),
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)
