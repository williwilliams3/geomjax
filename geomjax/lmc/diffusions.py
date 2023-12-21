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
"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from geomjax.types import ArrayTree
from jax.scipy.linalg import solve
from geomjax.util import generate_gaussian_noise
from jax.flatten_util import ravel_pytree

__all__ = ["overdamped_langevin_riemannian", "DiffusionState", "get_Gamma_fn"]


class DiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: ArrayTree
    Gamma: ArrayTree


def overdamped_langevin_riemannian(logdensity_grad_fn, metric_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logdensity_grad, metric, Gamma = state

        ndim = jnp.ndim(metric)

        if ndim == 1:
            noise = generate_gaussian_noise(rng_key, position)
            position = jax.tree_util.tree_map(
                lambda p, g, n: p
                + step_size * (g / metric)
                + step_size * Gamma
                + jnp.sqrt(2 * step_size) * n / jnp.sqrt(metric),
                position,
                logdensity_grad,
                noise,
            )

        else:
            mean_vector = jax.tree_util.tree_map(
                lambda p, g: p
                + step_size * solve(metric, g, assume_a="pos")
                + step_size * Gamma,
                position,
                logdensity_grad,
            )
            # Naive implementation
            # Better: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.from_precision.html
            inv_metric = jnp.linalg.inv(metric)
            inv_metric = 0.5 * (inv_metric + inv_metric.T)
            position = jax.tree_util.tree_map(
                lambda p, mu, Sigma: jax.random.multivariate_normal(
                    rng_key, mu, Sigma, dtype=p.dtype
                ),
                position,
                mean_vector,
                (2 * step_size) * inv_metric,
            )

        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        metric = metric_fn(position)
        Gamma = get_Gamma_fn(position, metric_fn)
        return DiffusionState(position, logdensity, logdensity_grad, metric, Gamma)

    return one_step


def get_Gamma_fn(position, metric_fn):
    position = ravel_pytree(position)[0]
    metric = metric_fn(position)
    ndim = jnp.ndim(metric)
    if ndim == 1:
        inv_metric_fn = lambda theta: 1 / metric_fn(theta)
        d_invg = jax.jacfwd(inv_metric_fn)(position)
        Gamma = jnp.diag(d_invg)
    else:
        inv_metric_fn = lambda theta: jnp.linalg.inv(metric_fn(theta))
        d_invg = jax.jacfwd(inv_metric_fn)(position)
        Gamma = jnp.diag(d_invg)
    return Gamma
