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


FixedPointSolver = Callable[
    [Callable[[ArrayLikeTree], tuple[ArrayTree, ArrayTree]], ArrayLikeTree],
    tuple[ArrayLikeTree, ArrayLikeTree, any],
]


class FixedPointIterationInfo(NamedTuple):
    success: bool
    norm: float
    iters: int


def solve_fixed_point_iteration(
    func: Callable[[ArrayLikeTree], tuple[ArrayLikeTree, ArrayLikeTree]],
    x0: ArrayLikeTree,
    *,
    convergence_tol: float = 1e-6,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm_fn: Callable[[ArrayLikeTree], float] = lambda x: jnp.max(jnp.abs(x)),
) -> tuple[ArrayLikeTree, ArrayLikeTree, FixedPointIterationInfo]:
    """Solve for x = func(x) using a fixed point iteration"""

    def compute_norm(x: ArrayLikeTree, xp: ArrayLikeTree) -> float:
        return norm_fn(ravel_pytree(jax.tree_util.tree_map(jnp.subtract, x, xp))[0])

    def cond_fn(args: tuple[int, ArrayLikeTree, ArrayLikeTree, float]) -> bool:
        n, _, _, norm = args
        return (
            (n < max_iters)
            & jnp.isfinite(norm)
            & (norm < divergence_tol)
            & (norm > convergence_tol)
        )

    def body_fn(
        args: tuple[int, ArrayLikeTree, ArrayLikeTree, float]
    ) -> tuple[int, ArrayLikeTree, ArrayLikeTree, float]:
        n, x, _, _ = args
        xn, aux = func(x)
        norm = compute_norm(xn, x)
        return n + 1, xn, aux, norm

    x, aux = func(x0)
    iters, x, aux, norm = jax.lax.while_loop(
        cond_fn, body_fn, (0, x, aux, compute_norm(x, x0))
    )
    success = jnp.isfinite(norm) & (norm <= convergence_tol)
    return x, aux, FixedPointIterationInfo(success, norm, iters)


def implicit_midpoint(
    logdensity_fn: Callable,
    kinetic_energy_fn: Callable,
    *,
    solver: FixedPointSolver = solve_fixed_point_iteration,
    **solver_kwargs: any,
) -> Callable[[IntegratorState, float], IntegratorState]:
    """The implicit midpoint integrator with support for non-stationary kinetic energy

    This is an integrator based on :cite:t:`brofos2021evaluating`, which provides
    support for kinetic energies that depend on position. This integrator requires that
    the kinetic energy function takes two arguments: position and momentum.

    The ``solver`` parameter allows overloading of the fixed point solver. By default, a
    simple fixed point iteration is used, but more advanced solvers could be implemented
    in the future.
    """
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    kinetic_energy_grad_fn = jax.grad(
        lambda q, p: kinetic_energy_fn(p, position=q), argnums=(0, 1)
    )

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, _ = state

        def _update(
            q: ArrayLikeTree,
            p: ArrayLikeTree,
            dUdq: ArrayLikeTree,
            initial: tuple[ArrayLikeTree, ArrayLikeTree] = (position, momentum),
        ) -> tuple[ArrayLikeTree, ArrayLikeTree]:
            dTdq, dHdp = kinetic_energy_grad_fn(q, p)
            dHdq = jax.tree_util.tree_map(jnp.subtract, dTdq, dUdq)

            # Take a step from the _initial coordinates_ using the gradients of the
            # Hamiltonian evaluated at the current guess for the midpoint
            q = jax.tree_util.tree_map(
                lambda q_, d_: q_ + 0.5 * step_size * d_, initial[0], dHdp
            )
            p = jax.tree_util.tree_map(
                lambda p_, d_: p_ - 0.5 * step_size * d_, initial[1], dHdq
            )
            return q, p

        # Solve for the midpoint numerically
        def _step(args: ArrayLikeTree) -> tuple[ArrayLikeTree, ArrayLikeTree]:
            q, p = args
            _, dLdq = logdensity_and_grad_fn(q)
            return _update(q, p, dLdq), dLdq

        (q, p), dLdq, info = solver(_step, (position, momentum), **solver_kwargs)
        del info  # TODO: Track the returned info

        # Take an explicit update as recommended by Brofos & Lederman
        _, dLdq = logdensity_and_grad_fn(q)
        q, p = _update(q, p, dLdq, initial=(q, p))

        return IntegratorState(q, p, *logdensity_and_grad_fn(q), 0.0)

    return one_step
