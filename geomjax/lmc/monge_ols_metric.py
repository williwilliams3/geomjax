import jax
from typing import Callable
from geomjax.types import ArrayLike, Array
import jax.numpy as jnp


# Adapt only alpha
def adapt_alpha2_ols(
    gradient_fn: Callable, fisher_metric_fn: Callable, enforce_possitivity: bool = True
) -> Callable:
    def adapt_fn(theta):
        dl = gradient_fn(theta)
        norm2_dl = jnp.dot(dl, dl)
        dl_normalized = dl / norm2_dl
        alpha2_hat = (
            jnp.dot(fisher_metric_fn(theta) @ dl_normalized, dl_normalized)
            - 1 / norm2_dl
        )
        alpha2_hat = jnp.clip(alpha2_hat, a_min=0.0)
        if enforce_possitivity:
            alpha2_hat = jnp.clip(alpha2_hat, a_min=0.0)
        return alpha2_hat, jnp.ones(len(theta))

    return adapt_fn


# Adapt alpha and real m
def adapt_alpha2_m_ols(
    gradient_fn: Callable, fisher_metric_fn: Callable, enforce_possitivity: bool = True
) -> Callable:
    def adapt_alpha_m_fn(theta):
        D = len(theta)
        dl = gradient_fn(theta)
        norm_dl = jnp.linalg.norm(dl)
        dl_normalized = dl / norm_dl
        F = fisher_metric_fn(theta)
        trF = jnp.trace(F)
        dlFdl = jnp.dot(F @ dl_normalized, dl_normalized)

        m_hat = (trF - dlFdl) / (D - 1)
        alpha2_hat = (D * dlFdl - trF) / ((D - 1) * norm_dl**2)
        if enforce_possitivity:
            diagm_hat = jnp.clip(diagm_hat, a_min=1e-6)  # Small identity term
            alpha2_hat = jnp.clip(alpha2_hat, a_min=0.0)
        return alpha2_hat, m_hat * jnp.ones(len(theta))

    return adapt_alpha_m_fn


# Adapt alpha and diag(m)
def adapt_alpha2_diagm_ols(
    gradient_fn: Callable, fisher_metric_fn: Callable, enforce_possitivity=True
) -> Callable:
    def adapt_alpha_diagm_fn(theta):
        dl = gradient_fn(theta)
        norm2_dl = jnp.dot(dl, dl)
        F = fisher_metric_fn(theta)
        a = dl**2
        c = jnp.diag(F)
        k = norm2_dl**2 - jnp.dot(a, a)
        term1 = jnp.dot(c, a / k)
        term2 = jnp.dot(F @ dl, dl / k)
        diagm_hat = c + (term1 - term2) * a
        alpha2_hat = term2 - term1
        if enforce_possitivity:
            diagm_hat = jnp.clip(diagm_hat, a_min=1e-6)  # Small identity term
            alpha2_hat = jnp.clip(alpha2_hat, a_min=0.0)
        return alpha2_hat, diagm_hat

    return adapt_alpha_diagm_fn


def monge_ols(
    logdensity_fn: Callable,
    metric_fn: Callable,
    option: str = "adapt-alpha-diagm",
    enforce_possitivity: bool = True,
) -> Callable:
    """
    Adaptation of parameters of monge metric by OLS to Fisher metric.
    input:
        logdensity_fn: log density
        metric_fn: assumed to be DxD
    Returns: Callable metric function
        enforce_possitivity: restrics solutions to be possitive definite

    The code constructs a full DxD matrix, which is not efficient in nature.
    Alternatively we can derive an Integrator for the metric which reduces
    all operations to vectors and vector products with closed form inverses.
    """
    gradient_fn = jax.grad(logdensity_fn)

    def metric(theta: ArrayLike) -> Array:
        dl = gradient_fn(theta)
        alpha2, diagm = adapt_fn(theta)
        metric = jnp.diag(diagm) + alpha2 * jnp.outer(dl, dl)
        return 0.5 * (metric + metric.T)

    if option == "monge":
        adapt_fn = lambda theta: (1, jnp.ones(len(theta)))

    elif option == "adapt-alpha":
        adapt_fn = adapt_alpha2_ols(
            gradient_fn, metric_fn, enforce_possitivity=enforce_possitivity
        )

    elif option == "adapt-alpha-m":
        adapt_fn = adapt_alpha2_m_ols(
            gradient_fn, metric_fn, enforce_possitivity=enforce_possitivity
        )

    elif option == "adapt-alpha-diagm":
        adapt_fn = adapt_alpha2_diagm_ols(
            gradient_fn, metric_fn, enforce_possitivity=enforce_possitivity
        )

    return metric
