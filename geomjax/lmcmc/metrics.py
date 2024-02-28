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
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.stats as jss
from jax.flatten_util import ravel_pytree
from geomjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from geomjax.util import generate_gaussian_noise
from typing import Optional

__all__ = ["riemannian_euclidean"]

RiemannianKineticEnergy = Callable[[ArrayLikeTree], float]


def gaussian_riemannian(
    metric_fn: Callable[[ArrayLikeTree], Array]
) -> tuple[Callable, RiemannianKineticEnergy, Callable]:
    r"""Hamiltonian dynamic on euclidean manifold with normally-distributed momentum :cite:p:`betancourt2013general`.

    The gaussian euclidean metric is a euclidean metric further characterized
    by setting the conditional probability density :math:`\pi(momentum|position)`
    to follow a standard gaussian distribution. A Newtonian hamiltonian
    dynamics is assumed.

    Parameters
    ----------
    metric_fn
        One or two-dimensional function corresponding respectively to a diagonal
        or full metric tensor.

    Returns
    -------
    velocity_generator
        A function that generates a value for the momentum at random.
    kinetic_energy
        A function that returns the kinetic energy given the momentum.
    is_turning
        A function that determines whether a trajectory is turning back on
        itself given the values of the momentum along the trajectory.
    omega_tilde_fn
        Omega tilde function.
    grad_logdetmetric
        Gradient of the log determinant of the metric.
    metric_vector_product
        Metric vector product.
    """

    def velocity_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        position, _ = ravel_pytree(position)
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)  # type: ignore[arg-type]
        shape = jnp.shape(metric)[:1]  # type: ignore[arg-type]
        metric = 0.5 * (metric + metric.T)
        if ndim == 1:  # diagonal mass matrix
            metric_invsqrt = 1 / jnp.sqrt(metric)
        elif ndim == 2:
            # inverse mass matrix can be factored into L*L.T. We want the cholesky
            # factor (inverse of L.T) of the mass matrix.
            L = jscipy.linalg.cholesky(metric, lower=True)
            identity = jnp.identity(shape[0])
            metric_invsqrt = jscipy.linalg.solve_triangular(
                L, identity, lower=True, trans=True
            )
        return generate_gaussian_noise(rng_key, position, sigma=metric_invsqrt)

    def kinetic_energy(
        position: ArrayLikeTree,
        velocity: ArrayLikeTree,
    ) -> float:
        position, _ = ravel_pytree(position)
        velocity, _ = ravel_pytree(velocity)
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)  # type: ignore[arg-type]
        if ndim == 1:  # diagonal mass matrix
            logdetG = jnp.sum(jnp.log(metric))
            kinetic_energy_val = -0.5 * logdetG + 0.5 * jnp.dot(
                metric * velocity, velocity
            )
        elif ndim == 2:
            metric = 0.5 * (metric + metric.T)
            _, logdetG = jnp.linalg.slogdet(metric)
            kinetic_energy_val = -0.5 * logdetG + 0.5 * jnp.dot(
                jnp.matmul(metric, velocity), velocity
            )
        return kinetic_energy_val

    def is_turning(
        velocity_left: ArrayLikeTree,
        velocity_right: ArrayLikeTree,
        velocity_sum: ArrayLikeTree,
    ) -> bool:
        """Generalized U-turn criterion :cite:p:`betancourt2013generalizing,nuts_uturn`.

        Parameters
        ----------
        momentum_left
            Momentum of the leftmost point of the trajectory.
        momentum_right
            Momentum of the rightmost point of the trajectory.
        momentum_sum
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

    def omega_tilde_fn(
        position: ArrayLikeTree,
        velocity: ArrayLikeTree,
        step_size: float,
    ) -> ArrayTree:
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)
        d_g = jax.jacfwd(metric_fn)(position)
        if ndim == 1:
            # Einstein summation
            partial_1 = jnp.diag(jnp.dot(d_g, velocity))
            partial_2 = d_g * velocity[:, None]
            Omega_tilde = 0.5 * (partial_1 + partial_2 - partial_2.T)
            result = jnp.diag(metric) + 0.5 * step_size * Omega_tilde
        else:
            # Einstein summation
            partial_1 = jnp.einsum("i,jli->lj", velocity, d_g)
            partial_2 = jnp.einsum("i,ilj->lj", velocity, d_g)
            partial_3 = jnp.einsum("i,ijl->lj", velocity, d_g)
            Omega_tilde = 0.5 * (partial_1 + partial_2 - partial_3)
            result = metric + 0.5 * step_size * Omega_tilde
        return result

    def grad_logdetmetric(position: ArrayLikeTree) -> ArrayTree:
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)
        if ndim == 1:  # diagonal mass matrix
            logdet_metric_fn = lambda theta: jnp.sum(jnp.log(metric_fn(theta)))
        else:
            logdet_metric_fn = lambda theta: jnp.linalg.slogdet(metric_fn(theta))[1]
        grad_logdet_metric = jax.grad(logdet_metric_fn)(position)
        return grad_logdet_metric

    def metric_vector_product(position: ArrayLikeTree, velocity: ArrayLikeTree):
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)
        if ndim == 1:  # diagonal mass matrix
            return jnp.multiply(metric, velocity)
        else:
            return jnp.dot(metric_fn(position), velocity)

    return (
        velocity_generator,
        kinetic_energy,
        is_turning,
        omega_tilde_fn,
        grad_logdetmetric,
        metric_vector_product,
    )
