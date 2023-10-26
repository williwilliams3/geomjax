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
from geomjax.types import ArrayLikeTree, ArrayTree
from geomjax.util import hvp

__all__ = ["mclachlan", "velocity_verlet", "yoshida", "lan_integrator"]


class RiemannianIntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the logdensity function (negative potential energy)
    to speedup computations.
    """

    alpha2: float
    position: ArrayTree
    velocity: ArrayTree
    logdensity: float
    logdensity_grad_norm: ArrayTree
    dl_ig: ArrayTree
    Hdl_ig: ArrayTree
    ig_Hdl_ig: ArrayTree
    logdensity_hvp_velocity_norm: ArrayTree
    determinant_metric: float
    volume_adjustment: float


RiemannianIntegrator = Callable[
    [RiemannianIntegratorState, float], RiemannianIntegratorState
]


def lan_integrator(
    logdensity_fn: Callable,
    set_weighted_gradient: Callable,
    normalizing_constant: Callable,
) -> RiemannianIntegrator:
    """Lans integrator ."""

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    hvp_fn = lambda x, v: hvp(logdensity_fn, x, v)

    def half_step_omegatilde(
        alpha2: float,
        v: ArrayLikeTree,
        J: float,
        dl: ArrayLikeTree,
        Hv: ArrayLikeTree,
        L: float,
        dl_ig: ArrayLikeTree,
        Hdl_ig: ArrayLikeTree,
        ig_Hdl_ig: ArrayLikeTree,
        eps: float,
    ) -> tuple[ArrayTree, float]:
        v, unravel_fn = ravel_pytree(v)
        dl, _ = ravel_pytree(dl)
        Hv, _ = ravel_pytree(Hv)
        dl_ig, _ = ravel_pytree(dl_ig)
        Hdl_ig, _ = ravel_pytree(Hdl_ig)
        ig_Hdl_ig, _ = ravel_pytree(ig_Hdl_ig)

        # 1st volume adustment
        determinant_first_volume_adjustment = 1.0 + 0.5 * eps * alpha2 * jnp.dot(
            Hv, dl_ig
        )
        J -= jnp.log(jnp.abs(determinant_first_volume_adjustment))

        # half velocity update
        v += (
            alpha2 * L * jnp.dot(dl, v) + 0.5 * eps * jnp.sqrt(L)
        ) * dl_ig - 0.5 * alpha2 * eps * ig_Hdl_ig
        numerator = alpha2 * (jnp.dot(dl, v) + 0.5 * eps * jnp.dot(Hv, v))
        v -= numerator / determinant_first_volume_adjustment * dl_ig

        # 2nd volume adustment
        J += jnp.log(jnp.abs(1.0 - 0.5 * eps * alpha2 * jnp.dot(Hdl_ig, v)))

        return unravel_fn(v), J

    def half_step_omega(
        alpha2: float,
        v: ArrayLikeTree,
        J: float,
        dl: ArrayLikeTree,
        Hv: ArrayLikeTree,
        L: float,
        dl_ig: ArrayLikeTree,
        Hdl_ig: ArrayLikeTree,
        ig_Hdl_ig: ArrayLikeTree,
        eps: float,
    ) -> tuple[ArrayTree, float]:
        v, unravel_fn = ravel_pytree(v)
        dl, _ = ravel_pytree(dl)
        Hv, _ = ravel_pytree(Hv)
        dl_ig, _ = ravel_pytree(dl_ig)
        Hdl_ig, _ = ravel_pytree(Hdl_ig)
        ig_Hdl_ig, _ = ravel_pytree(ig_Hdl_ig)

        # 1st volume adustment
        determinant_first_volume_adjustment = 1.0 + 0.5 * eps * alpha2 * jnp.dot(
            Hv, dl_ig
        )
        J -= jnp.log(jnp.abs(determinant_first_volume_adjustment))

        # Velocity update
        sL = jnp.sqrt(L)
        dphi = -dl + alpha2 * Hdl_ig / sL
        dphi_ig = -dl_ig + alpha2 * ig_Hdl_ig / sL
        v_temp = v - 0.5 * eps * sL * (dphi_ig - alpha2 * jnp.dot(dphi, dl_ig) * dl_ig)
        v_update = (
            v_temp
            - 0.5
            * eps
            * jnp.dot(v_temp, Hv)
            / determinant_first_volume_adjustment
            * dl_ig
        )

        # 2nd volume adustment
        J += jnp.log(jnp.abs(1.0 - 0.5 * eps * alpha2 * jnp.dot(Hdl_ig, v_update)))

        return unravel_fn(v_update), J

    def one_step(
        state: RiemannianIntegratorState,
        step_size: float,
        half_step_fn=half_step_omega,
    ) -> RiemannianIntegratorState:
        (
            alpha2,
            position,
            velocity,
            logdensity,
            logdensity_grad_norm,
            dl_ig,
            Hdl_ig,
            ig_Hdl_ig,
            logdensity_hvp_velocity_norm,
            determinant_metric,
            volume_adjustment,
        ) = state

        # 1st half step
        velocity, volume_adjustment = half_step_fn(
            alpha2,
            velocity,
            volume_adjustment,
            logdensity_grad_norm,
            logdensity_hvp_velocity_norm,
            determinant_metric,
            dl_ig,
            Hdl_ig,
            ig_Hdl_ig,
            step_size,
        )

        position = jax.tree_util.tree_map(
            lambda position, velocity: position + step_size * velocity,
            position,
            velocity,
        )

        # update derivatives, determinant and hessian vector product
        logdensity, logdensity_grad = logdensity_and_grad_fn(position)
        determinant_metric = normalizing_constant(alpha2, logdensity_grad)
        sqrt_determinant_metric = jnp.sqrt(determinant_metric)
        # normalized quantities
        logdensity_grad_norm = logdensity_grad / sqrt_determinant_metric
        logdensity_hvp_velocity_norm = (
            hvp_fn(position, velocity) / sqrt_determinant_metric
        )

        # Quantities multiplied by the metric
        dl_ig = set_weighted_gradient(logdensity_grad_norm)
        Hdl_ig = hvp_fn(position, dl_ig) / sqrt_determinant_metric
        ig_Hdl_ig = set_weighted_gradient(Hdl_ig)

        # 2nd half step
        velocity, volume_adjustment = half_step_fn(
            alpha2,
            velocity,
            volume_adjustment,
            logdensity_grad_norm,
            logdensity_hvp_velocity_norm,
            determinant_metric,
            dl_ig,
            Hdl_ig,
            ig_Hdl_ig,
            step_size,
        )
        # Update hessian vector product
        logdensity_hvp_velocity_norm = (
            hvp_fn(position, velocity) / sqrt_determinant_metric
        )

        return RiemannianIntegratorState(
            alpha2,
            position,
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

    return one_step


def lan_integrator_general(
    logdensity_fn: Callable,
    christoffel_product: Callable,
    grad_logdet_metric: Callable,
    inverse_metric_product: Callable,
) -> RiemannianIntegrator:
    """Lans integrator ."""

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)

    def half_step_fn(
        theta: ArrayLikeTree,
        v: ArrayLikeTree,
        J: float,
        dl: ArrayLikeTree,
        eps: float,
    ) -> tuple[ArrayTree, float]:
        v, unravel_fn = ravel_pytree(v)
        theta, _ = ravel_pytree(theta)
        dl, _ = ravel_pytree(dl)
        Hv, _ = ravel_pytree(Hv)
        dl_ig, _ = ravel_pytree(dl_ig)
        Hdl_ig, _ = ravel_pytree(Hdl_ig)
        ig_Hdl_ig, _ = ravel_pytree(ig_Hdl_ig)

        # Auxiliary quantities
        Omega = christoffel_product(theta, v)
        dhalflogdetmet = 0.5 * grad_logdet_metric(theta)

        # 1st volume adustment
        Omega_term = jnp.eye(len(v)) + 0.5 * eps * Omega
        J -= jnp.log(jnp.abs(jnp.linalg.det(Omega_term)))

        # half velocity update
        dphi = -dl + dhalflogdetmet
        v_temp = v - 0.5 * eps * inverse_metric_product(theta, dphi)
        v_update = jnp.linalg.solve(Omega_term, v_temp)

        # 2nd volume adustment
        Omega = christoffel_product(theta, v_update)
        J += jnp.log(jnp.abs(jnp.linalg.det(jnp.eye(len(v)) - 0.5 * eps * Omega)))

        return unravel_fn(v_update), J

    def one_step(
        state: RiemannianIntegratorState,
        step_size: float,
    ) -> RiemannianIntegratorState:
        (
            _,
            position,
            velocity,
            logdensity,
            logdensity_grad,
            _,
            _,
            _,
            _,
            _,
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

        return RiemannianIntegratorState(
            _,
            position,
            velocity,
            logdensity,
            logdensity_grad,
            _,
            _,
            _,
            _,
            _,
            volume_adjustment,
        )

    return one_step
