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
    inverse_mass_matrix
        One or two-dimensional array corresponding respectively to a diagonal
        or dense mass matrix. The inverse mass matrix is multiplied to a
        flattened version of the Pytree in which the chain position is stored
        (the current value of the random variables). The order of the variables
        should thus match JAX's tree flattening order, and more specifically
        that of `ravel_pytree`.
        In particular, JAX sorts dictionaries by key when flattening them. The
        value of each variables will appear in the flattened Pytree following
        the order given by `sort(keys)`.

    Returns
    -------
    momentum_generator
        A function that generates a value for the momentum at random.
    kinetic_energy
        A function that returns the kinetic energy given the momentum.
    is_turning
        A function that determines whether a trajectory is turning back on
        itself given the values of the momentum along the trajectory.

    """

    def velocity_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        position, _ = ravel_pytree(position)
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)  # type: ignore[arg-type]
        shape = jnp.shape(metric)[:1]  # type: ignore[arg-type]
        metric = 0.5 * (metric + metric.T)
        if ndim == 1:  # diagonal mass matrix
            metric_sqrt = jnp.sqrt(jnp.reciprocal(metric))
        elif ndim == 2:
            # inverse mass matrix can be factored into L*L.T. We want the cholesky
            # factor (inverse of L.T) of the mass matrix.
            L = jscipy.linalg.cholesky(metric, lower=True)
            identity = jnp.identity(shape[0])
            metric_sqrt = jscipy.linalg.solve_triangular(
                L, identity, lower=True, trans=True
            )
        return generate_gaussian_noise(rng_key, position, sigma=metric_sqrt)

    def kinetic_energy(
        position: ArrayLikeTree,
        velocity: ArrayLikeTree,
    ) -> float:
        position, _ = ravel_pytree(position)
        velocity, _ = ravel_pytree(velocity)
        metric = metric_fn(position)
        ndim = jnp.ndim(metric)  # type: ignore[arg-type]
        metric = 0.5 * (metric + metric.T)
        if ndim == 1:  # diagonal mass matrix
            logdetG = jnp.sum(jnp.log(metric))
        elif ndim == 2:
            _, logdetG = jnp.linalg.slogdet(metric)
        kinetic_energy_val = -0.5 * logdetG + 0.5 * jnp.dot(metric @ velocity, velocity)
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
            partial_1 = jnp.einsum("i,il->l", velocity, d_g)
            partial_3 = jnp.einsum("i,li->l", velocity, d_g)
            Omega_tilde = partial_1 - 0.5 * partial_3
        else:
            # Einstein summation
            partial_1 = jnp.einsum("i,jli->lj", velocity, d_g)
            partial_2 = jnp.einsum("i,ilj->lj", velocity, d_g)
            partial_3 = jnp.einsum("i,ijl->lj", velocity, d_g)
            Omega_tilde = 0.5 * (partial_1 + partial_2 - partial_3)
        return metric + 0.5 * step_size * Omega_tilde

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


def gaussian_riemannian_mommentum(
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
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: Optional[ArrayLikeTree] = None,
        position_right: Optional[ArrayLikeTree] = None,
    ) -> bool:
        del momentum_left, momentum_right, momentum_sum, position_left, position_right
        raise NotImplementedError(
            "NUTS sampling is not yet implemented for Riemannian manifolds"
        )

        # Here's a possible implementation of this function, but the NUTS
        # proposal will require some refactoring to work properly, since we need
        # to be able to access the coordinates at the left and right endpoints
        # to compute the mass matrix at those points.

        # m_left, _ = ravel_pytree(momentum_left)
        # m_right, _ = ravel_pytree(momentum_right)
        # m_sum, _ = ravel_pytree(momentum_sum)

        # mass_matrix_left = mass_matrix_fn(position_left)
        # mass_matrix_right = mass_matrix_fn(position_right)
        # velocity_left = jnp.linalg.solve(mass_matrix_left, m_left)
        # velocity_right = jnp.linalg.solve(mass_matrix_right, m_right)

        # # rho = m_sum
        # rho = m_sum - (m_right + m_left) / 2
        # turning_at_left = jnp.dot(velocity_left, rho) <= 0
        # turning_at_right = jnp.dot(velocity_right, rho) <= 0
        # return turning_at_left | turning_at_right

    return momentum_generator, kinetic_energy, is_turning
