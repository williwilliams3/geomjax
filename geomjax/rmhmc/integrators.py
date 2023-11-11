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
"""Implicit fixed point integrator for RMHMC."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from geomjax.types import ArrayTree, ArrayLikeTree

__all__ = ["implicit_midpoint"]


class IntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the logdensity function (negative potential energy)
    to speedup computations.
    """

    position: ArrayTree
    momentum: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


RiemannianIntegrator = Callable[[IntegratorState, float], IntegratorState]


def new_integrator_state(logdensity_fn, position, momentum):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(position, momentum, logdensity, logdensity_grad)


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
        lambda q, p: kinetic_energy_fn(momentum=p, position=q), argnums=(0, 1)
    )

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, *_ = state

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

        return IntegratorState(q, p, *logdensity_and_grad_fn(q))

    return one_step
