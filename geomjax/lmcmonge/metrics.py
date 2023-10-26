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
r"""Metric space in which the Hamiltonian dynamic is embedded.

An important particular case (and the most used in practice) of metric for the
position space in the Euclidean metric. It is defined by a definite positive
matrix :math:`M` with fixed value so that the kinetic energy of the hamiltonian
dynamic is independent of the position and only depends on the momentum
:math:`p` :cite:p:`betancourt2017geometric`.

For a Newtonian hamiltonian dynamic the kinetic energy is given by:

.. math::
    K(p) = \frac{1}{2} p^T M^{-1} p

We can also generate a relativistic dynamic :cite:p:`lu2017relativistic`.

"""
from typing import Callable

import jax.numpy as jnp
import jax.scipy as jscipy
from jax.flatten_util import ravel_pytree

from geomjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from geomjax.util import generate_gaussian_noise

__all__ = ["gaussian_euclidean", "riemannian_euclidean"]

EuclideanKineticEnergy = Callable[[ArrayLikeTree], float]


def gaussian_euclidean(
    inverse_mass_matrix: Array,
) -> tuple[Callable, EuclideanKineticEnergy, Callable]:
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
    ndim = jnp.ndim(inverse_mass_matrix)  # type: ignore[arg-type]
    shape = jnp.shape(inverse_mass_matrix)[:1]  # type: ignore[arg-type]

    if ndim == 1:  # diagonal mass matrix
        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))
        matmul = jnp.multiply

    elif ndim == 2:
        # inverse mass matrix can be factored into L*L.T. We want the cholesky
        # factor (inverse of L.T) of the mass matrix.
        L = jscipy.linalg.cholesky(inverse_mass_matrix, lower=True)
        identity = jnp.identity(shape[0])
        mass_matrix_sqrt = jscipy.linalg.solve_triangular(
            L, identity, lower=True, trans=True
        )
        # Note that mass_matrix_sqrt is a upper triangular matrix here, with
        #   jscipy.linalg.inv(mass_matrix_sqrt @ mass_matrix_sqrt.T) == inverse_mass_matrix
        # An alternative is to compute directly the cholesky factor of the inverse mass matrix
        #   mass_matrix_sqrt = jscipy.linalg.cholesky(jscipy.linalg.inv(inverse_mass_matrix), lower=True)
        # which the result would instead be a lower triangular matrix.
        matmul = jnp.matmul

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(inverse_mass_matrix)}."  # type: ignore[arg-type]
        )

    def momentum_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(momentum: ArrayLikeTree) -> float:
        momentum, _ = ravel_pytree(momentum)
        velocity = matmul(inverse_mass_matrix, momentum)
        kinetic_energy_val = 0.5 * jnp.dot(velocity, momentum)
        return kinetic_energy_val

    def is_turning(
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
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
        m_left, _ = ravel_pytree(momentum_left)
        m_right, _ = ravel_pytree(momentum_right)
        m_sum, _ = ravel_pytree(momentum_sum)

        velocity_left = matmul(inverse_mass_matrix, m_left)
        velocity_right = matmul(inverse_mass_matrix, m_right)

        # rho = m_sum
        rho = m_sum - (m_right + m_left) / 2
        turning_at_left = jnp.dot(velocity_left, rho) <= 0
        turning_at_right = jnp.dot(velocity_right, rho) <= 0
        return turning_at_left | turning_at_right

    return momentum_generator, kinetic_energy, is_turning


def set_inverse_mass(
    alpha2: float, diagonal_covariance: Array, do_power="fixed"
) -> tuple[Array, Array]:
    inverse_diagonal_mass = diagonal_covariance
    if do_power == "fixed":
        inverse_diagonal_mass = diagonal_covariance
    elif do_power == "power":
        inverse_diagonal_mass = diagonal_covariance ** (1 - alpha2)
    elif do_power == "inverse_convex":
        inverse_diagonal_mass = (1 - alpha2) * diagonal_covariance + alpha2
    elif do_power == "convex":
        diagonal_mass = (1 - alpha2) * jnp.reciprocal(diagonal_covariance) + alpha2 * 1
        inverse_diagonal_mass = jnp.reciprocal(diagonal_mass)
    else:
        raise Exception("options: fixed, power inverse_convex, convex, linear_increase")
    diagonal_mass = jnp.reciprocal(inverse_diagonal_mass)
    return inverse_diagonal_mass, diagonal_mass


def innerproduct_monge(
    alpha2: float,
    v: ArrayLikeTree,
    u: ArrayLikeTree,
    dl,
    diagonal_mass: Array,
    L: float,
) -> float:
    u_g = jnp.multiply(u, diagonal_mass)
    return jnp.dot(u_g, v) + alpha2 * L * jnp.dot(v, dl) * jnp.dot(u, dl)


def norm_monge(
    alpha2: float,
    v: ArrayLikeTree,
    dl: ArrayLikeTree,
    diagonal_mass: Array,
    L: float,
) -> float:
    return jnp.sqrt(innerproduct_monge(alpha2, v, v, dl, diagonal_mass, L))


def angle_monge(
    alpha2,
    u: ArrayLikeTree,
    v: ArrayLikeTree,
    dl: ArrayLikeTree,
    diagonal_mass: Array,
    L: float,
) -> float:
    # For numerical reasons, can exceed 1.0, or be less than -1, so we double clip
    # If a vector is zero, we do not account for it
    norm_u = norm_monge(alpha2, u, dl, diagonal_mass, L)
    norm_v = norm_monge(alpha2, v, dl, diagonal_mass, L)
    if (norm_u == 0) or (norm_v == 0):
        return jnp.arccos(1)
    else:
        inner_product = innerproduct_monge(u, v, dl) / (norm_u * norm_v)
        return jnp.arccos(jnp.clip(inner_product, -1.0, 1.0))


def gaussian_riemannian(
    alpha2: float,
    diagonal_covariance: Array,
) -> tuple[Callable, EuclideanKineticEnergy, Callable, Callable]:
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
    inverse_diagonal_mass, diagonal_mass = set_inverse_mass(alpha2, diagonal_covariance)

    ndim = jnp.ndim(inverse_diagonal_mass)  # type: ignore[arg-type]

    if ndim == 1:  # diagonal mass matrix
        matmul = jnp.multiply

    elif ndim == 2:
        # we assume diagonal inverse mass metric
        # the Riemannian metric is diag(g) + outer(alpha*dl, alpha*dl )
        matmul = jnp.matmul

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(inverse_diagonal_mass)}."  # type: ignore[arg-type]
        )

    def velocity_generator(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        alpha2: float,
        dl_ig: ArrayLikeTree,
    ) -> ArrayTree:
        dl_ig, _ = ravel_pytree(dl_ig)
        inverse_metric = jnp.diag(inverse_diagonal_mass) - alpha2 * jnp.outer(
            dl_ig, dl_ig
        )
        L = jscipy.linalg.cholesky(inverse_metric, lower=True)
        return generate_gaussian_noise(rng_key, position, sigma=L)

    def kinetic_energy(
        velocity: ArrayLikeTree,
        alpha2: float,
        logdensity_grad: ArrayLikeTree,
        determinant_metric: float,
    ) -> float:
        logdensity_grad, _ = ravel_pytree(logdensity_grad)
        velocity, _ = ravel_pytree(velocity)
        L = determinant_metric
        g_sqrt = jnp.sqrt(diagonal_mass)
        v_g = jnp.multiply(g_sqrt, velocity)
        Energy = (
            # determinant with opposite sign from inverse
            -0.5 * (jnp.log(L) + jnp.sum(jnp.log(diagonal_mass)))
            # metric norm of velocity
            + 0.5 * jnp.dot(v_g, v_g)
            + 0.5 * L * alpha2 * jnp.dot(velocity, logdensity_grad) ** 2.0
        )
        return Energy

    def set_weighted_gradient(logdensity_grad: ArrayLikeTree) -> ArrayTree:
        # set the weighted gradient and Hessian vector product
        logdensity_grad, unravel_fn = ravel_pytree(logdensity_grad)
        return unravel_fn(matmul(inverse_diagonal_mass, logdensity_grad))

    def normalizing_constant(alpha2: float, logdensity_grad: ArrayLikeTree) -> float:
        # set the weighted gradient and Hessian vector product
        logdensity_grad, _ = ravel_pytree(logdensity_grad)
        logdensity_grad_weighted = matmul(inverse_diagonal_mass, logdensity_grad)
        return 1 + alpha2 * jnp.dot(logdensity_grad_weighted, logdensity_grad)

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

    def is_turning_riemannian(
        velocity_left: ArrayLikeTree,
        velocity_right: ArrayLikeTree,
        velocity_sum_left: ArrayLikeTree,
        velocity_sum_right: ArrayLikeTree,
        dl_left: ArrayLikeTree,
        dl_right: ArrayLikeTree,
        L_left: float,
        L_right: float,
        option="riemannian2",
    ) -> bool:
        """Geodesic U-turn criterion.

        Parameters
        ----------
        momentum_left
            Momentum of the leftmost point of the trajectory.
        momentum_right
            Momentum of the rightmost point of the trajectory.
        momentum_sum_left
            Sum of the momenta leftmost point of the trajectory.
        momentum_sum_right
            Sum of the momenta rightmost point of the trajectory.

        """
        velocity_left, _ = ravel_pytree(velocity_left)
        velocity_left, _ = ravel_pytree(velocity_right)
        velocity_sum_left, _ = ravel_pytree(velocity_sum_left)
        velocity_sum_right, _ = ravel_pytree(velocity_sum_right)
        if option == "euclidean":
            rho = velocity_sum_left + velocity_sum_right
            turning_at_left = jnp.dot(velocity_left, rho) <= 0
            turning_at_right = jnp.dot(velocity_right, rho) <= 0
            return turning_at_left | turning_at_right
        if option == "riemannian1":
            rho = velocity_sum_left + velocity_sum_right
            turning_at_left = (
                innerproduct_monge(
                    alpha2, velocity_left, rho, dl_left, diagonal_mass, L_left
                )
                <= 0
            )
            turning_at_right = (
                innerproduct_monge(
                    alpha2, velocity_right, rho, dl_right, diagonal_mass, L_right
                )
                <= 0
            )
            return turning_at_left | turning_at_right
        if option == "riemannian2":
            turning_at_left = (
                innerproduct_monge(
                    alpha2,
                    velocity_left,
                    velocity_sum_left,
                    dl_left,
                    diagonal_mass,
                    L_left,
                )
                <= 0
            )
            turning_at_right = (
                innerproduct_monge(
                    alpha2,
                    velocity_right,
                    velocity_sum_right,
                    dl_right,
                    diagonal_mass,
                    L_right,
                )
                <= 0
            )
            angle1 = angle_monge(
                alpha2, velocity_left, velocity_sum_left, dl_left, diagonal_mass, L_left
            )
            angle2 = angle_monge(
                alpha2, velocity_right, rho, dl_right, diagonal_mass, L_right
            )
            turning_full = angle1 + angle2 > jnp.pi / 2
            return turning_at_left | turning_at_right | turning_full

    return (
        velocity_generator,
        kinetic_energy,
        set_weighted_gradient,
        normalizing_constant,
        is_turning,
    )
