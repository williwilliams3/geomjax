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
from geomjax.util import generate_gaussian_noise
from jax.scipy.linalg import cholesky, solve

__all__ = ["overdamped_langevin"]


class DiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


def overdamped_langevin_riemannian(logdensity_grad_fn, metric_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logdensity_grad = state
        noise = generate_gaussian_noise(rng_key, position)

        metric = metric_fn(position)
        ndim = jnp.ndim(metric)
        d_g = jax.jacfwd(metric_fn)(position)
        if ndim == 1:
            Gamma = jnp.diag(d_g)
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
            chol_factor = cholesky(metric, lower=True)
            position = jax.tree_util.tree_map(
                lambda p, g, n: p
                + step_size * solve(metric, g, assume_a=True)
                + step_size * Gamma
                + jnp.sqrt(2 * step_size) * solve(chol_factor, n),
                position,
                logdensity_grad,
                noise,
            )

        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        return DiffusionState(position, logdensity, logdensity_grad)

    return one_step
