import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import geomjax

# import matplotlib.pyplot as plt


def inference_loop(rng_key, sampler, initial_position, num_samples):
    initial_state = sampler.init(initial_position)

    kernel = sampler.step

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


class neal_funnel:
    def __init__(self, D=2, mean=0.0, sigma=3.0):
        self.sigma = sigma
        self.mean = mean
        self.D = D
        self.name = "NealFunnel"
        self.xlim = [-10.0, 10.0]
        self.ylim = [-10.0, 10.0]

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
    alpha2 = 1.0
    sampler = geomjax.lmcmonge(
        logdensity_fn,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
        alpha2=alpha2,
    )

    # Initialize the state
    initial_position = jnp.ones(M.D)
    # Sample 1 chain
    states = inference_loop(
        rng_key,
        sampler,
        initial_position,
        num_samples=1000,
    )
    # plt.scatter(states.position[:, 0], states.position[:, 1])
    # plt.grid()
    # plt.show()

    ##############
    # LMC Fisher
    sampler = geomjax.lmc(logdensity_fn, step_size, metric_fn, num_integration_steps)
    # Sample 1 chain
    states = inference_loop(
        rng_key,
        sampler,
        initial_position,
        num_samples=1000,
    )
    # plt.scatter(states.position[:, 0], states.position[:, 1])
    # plt.grid()
    # plt.show()
