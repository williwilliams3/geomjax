import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import geomjax


def inference_loop_multiple_chains(
    rng_key, sampler, initial_position, num_samples, num_chains
):
    # Assume all chains start at same possition
    kernel = sampler.step
    initial_positions = jnp.tile(initial_position, (num_chains, 1))
    initial_states = jax.vmap(sampler.init, in_axes=(0))(initial_positions)
    keys = jax.random.split(rng_key, num_chains)

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_states, keys)

    return states


class neal_funnel:
    def __init__(self, D=2, mean=0.0, sigma=3.0):
        self.sigma = sigma
        self.mean = mean
        self.D = D

    def logp(self, theta):
        return jss.norm.logpdf(theta[self.D - 1], loc=0.0, scale=self.sigma) + jnp.sum(
            jss.norm.logpdf(
                theta[: self.D - 1], loc=0.0, scale=jnp.exp(0.5 * theta[self.D - 1])
            )
        )

    def inverse_jacobian(self, theta):
        D = self.D
        upper_rows = jnp.c_[
            jnp.exp(-0.5 * theta[-1]) * jnp.eye(D - 1),
            -0.5 * jnp.exp(-0.5 * theta[-1]) * theta[0 : (D - 1)],
        ]
        lowest_row = jnp.append(jnp.zeros(D - 1), 1.0 / self.sigma)
        inverse_jacobian = jnp.r_[upper_rows, [lowest_row]]
        return inverse_jacobian

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)


if __name__ == "__main__":
    # Build the kernel
    # LMC Monge
    rng_key = jax.random.PRNGKey(0)
    M = neal_funnel()
    logdensity_fn = M.logp
    metric_fn = M.fisher_metric_fn
    step_size = 1e-1
    inverse_mass_matrix = jnp.ones(M.D)
    num_integration_steps = 8
    num_samples = 1000
    num_chains = 8
    initial_position = jnp.ones(2)
    ##############
    # LMC Fisher
    sampler = geomjax.lmc(logdensity_fn, step_size, metric_fn, num_integration_steps)
    # Sample chains
    states = inference_loop_multiple_chains(
        rng_key, sampler, initial_position, num_samples, num_chains
    )
    rhat = geomjax.rhat(states.position, chain_axis=1, sample_axis=0)
    ess = geomjax.ess(states.position, chain_axis=1, sample_axis=0)

    print(f"ESS {ess} and Rhat {rhat}")
