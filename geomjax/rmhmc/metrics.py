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
r"""Metric space in which the Lagrangian dynamic is embedded.

An important particular case of metric for the
position space in the Riemannian metric. It is defined by a definite positive
matrix :math:`G(theta)` with so that the kinetic energy of the Lagrangian
dynamic is independent of the position and only depends on the velocity
:math:`v` :cite:p:`lan2012lagrangian`.

For a Newtonian hamiltonian dynamic the kinetic energy is given by:

.. math::
    K(theta, v) = \frac{1}{2} v^T G(theta) v - 0.5* \log \det G(theta)


"""
from typing import Callable
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.stats as jss
from jax.flatten_util import ravel_pytree
from geomjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from geomjax.util import generate_gaussian_noise

__all__ = ["riemannian_euclidean"]

RiemannianKineticEnergy = Callable[[ArrayLikeTree], float]


def gaussian_riemannian(
    metric_fn: Callable[[ArrayLikeTree], Array],
) -> tuple[Callable, RiemannianKineticEnergy, Callable]:
    def momentum_generator(rng_key: PRNGKey, position: ArrayTree) -> ArrayTree:
        mass_matrix = metric_fn(position)
        ndim = jnp.ndim(mass_matrix)
        if ndim == 1:
            mass_matrix_sqrt = jnp.sqrt(mass_matrix)
        elif ndim == 2:
            mass_matrix_sqrt = jscipy.linalg.cholesky(mass_matrix, lower=True)
        else:
            raise ValueError(
                "The mass matrix has the wrong number of dimensions:"
                f" expected 1 or 2, got {jnp.ndim(mass_matrix)}."
            )

        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(position: ArrayLikeTree, momentum: ArrayLikeTree) -> float:
        momentum, _ = ravel_pytree(momentum)
        mass_matrix = metric_fn(position)
        ndim = jnp.ndim(mass_matrix)
        if ndim == 1:
            return -jnp.sum(jss.norm.logpdf(momentum, 0.0, jnp.sqrt(mass_matrix)))
        elif ndim == 2:
            return -jss.multivariate_normal.logpdf(
                momentum, jnp.zeros_like(momentum), mass_matrix
            )
        else:
            raise ValueError(
                "The mass matrix has the wrong number of dimensions:"
                f" expected 1 or 2, got {jnp.ndim(mass_matrix)}."
            )

    def is_turning(
        velocity_left: ArrayLikeTree,
        velocity_right: ArrayLikeTree,
        velocity_sum: ArrayLikeTree,
    ) -> bool:
        """Generalized U-turn criterion :cite:p:`betancourt2013generalizing,nuts_uturn`.

        Parameters
        ----------
        velocity_left
            Velocity of the leftmost point of the trajectory.
        velocity_right
            Velocity of the rightmost point of the trajectory.
        velocity_sum
            Sum of the momenta along the trajectory.

        """
        velocity_left, _ = ravel_pytree(velocity_left)
        velocity_right, _ = ravel_pytree(velocity_right)
        velocity_sum, _ = ravel_pytree(velocity_sum)

        rho = velocity_sum
        # rho = velocity_sum - (velocity_right + velocity_left) / 2
        turning_at_left = jnp.dot(velocity_left, rho) <= 0
        turning_at_right = jnp.dot(velocity_right, rho) <= 0
        return turning_at_left | turning_at_right

    def inverse_metric_vector_product(position: ArrayLikeTree, momentum: ArrayLikeTree):
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)
        if ndim == 1:  # diagonal mass matrix
            velocity = jnp.multiply(1 / metric, momentum)
        else:
            velocity = jnp.linalg.solve(metric_fn(position), momentum)
        return velocity

    return momentum_generator, kinetic_energy, is_turning, inverse_metric_vector_product
