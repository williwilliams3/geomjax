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
"""Symplectic, time-reversible, integrators for Hamiltonian trajectories."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from geomjax.types import ArrayTree, ArrayLikeTree
from jax.scipy.linalg import lu_factor, lu_solve

__all__ = ["lan_integrator"]


class IntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the logdensity function (negative potential energy)
    to speedup computations.
    """

    position: ArrayTree
    velocity: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    volume_adjustment: float


RiemannianIntegrator = Callable[[IntegratorState, float], IntegratorState]


def new_integrator_state(logdensity_fn, position, velocity, volume_adjustment):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(
        position, velocity, logdensity, logdensity_grad, volume_adjustment
    )


def lan_integrator(
    logdensity_fn: Callable,
    omega_tilde_fn: Callable,
    grad_logdetmetric: Callable,
    metric_vector_product: Callable,
) -> RiemannianIntegrator:
    """Lans integrator ."""

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)

    def half_step_fn(
        position: ArrayLikeTree,
        velocity: ArrayLikeTree,
        volume_adjustment: float,
        logdensity_grad: ArrayLikeTree,
        step_size: float,
    ) -> tuple[ArrayTree, float]:
        velocity, unravel_fn = ravel_pytree(velocity)
        position, _ = ravel_pytree(position)
        logdensity_grad, _ = ravel_pytree(logdensity_grad)

        # 1st volume adjustment
        Omega_tilde_term = omega_tilde_fn(position, velocity, step_size)
        lu, piv = lu_factor(Omega_tilde_term)
        logdet_Omega_tilde = jnp.sum(jnp.log(jnp.abs(jnp.diag(lu))))

        volume_adjustment -= logdet_Omega_tilde

        # Half velocity update
        dphi = -logdensity_grad + 0.5 * grad_logdetmetric(position)
        v_temp = metric_vector_product(position, velocity) - 0.5 * step_size * dphi
        v_update = lu_solve((lu, piv), v_temp)

        # 2nd volume adjustment
        Omega_tilde_term = omega_tilde_fn(position, v_update, -step_size)
        lu, piv = lu_factor(Omega_tilde_term)
        logdet_Omega_tilde = jnp.sum(jnp.log(jnp.abs(jnp.diag(lu))))

        volume_adjustment += logdet_Omega_tilde

        return unravel_fn(v_update), volume_adjustment

    def one_step(
        state: IntegratorState,
        step_size: float,
    ) -> IntegratorState:
        (
            position,
            velocity,
            logdensity,
            logdensity_grad,
            volume_adjustment,
        ) = state

        # 1st half step
        velocity, volume_adjustment = half_step_fn(
            position,
            velocity,
            volume_adjustment,
            logdensity_grad,
            step_size,
        )

        position = jax.tree_util.tree_map(
            lambda position, velocity: position + step_size * velocity,
            position,
            velocity,
        )

        # update derivatives, determinant and hessian vector product
        logdensity, logdensity_grad = logdensity_and_grad_fn(position)

        # 2nd half step
        velocity, volume_adjustment = half_step_fn(
            position,
            velocity,
            volume_adjustment,
            logdensity_grad,
            step_size,
        )

        return IntegratorState(
            position,
            velocity,
            logdensity,
            logdensity_grad,
            volume_adjustment,
        )

    return one_step
