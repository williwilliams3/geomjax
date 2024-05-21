import geomjax
import postjax
import jax.numpy as jnp
import jax.random as jr

rng_key = jr.key(42)
model = postjax.neal_funnel()
logdensity_fn = model.logp

sampler = geomjax.nutsrmhmc(
    logdensity_fn,
    step_size=1e-2,
    metric_fn=model.fisher_metric_fn,
)

state = sampler.init(jnp.zeros(2))
state, info = sampler.step(rng_key, state)

assert (state.position == jnp.array([-0.73879963, 1.2370402])).all()

sampler1 = geomjax.rmhmc(
    logdensity_fn,
    step_size=1e-2,
    num_integration_steps=10,
    metric_fn=lambda x: jnp.eye(2),
)
sampler2 = geomjax.hmc(
    logdensity_fn,
    step_size=1e-2,
    num_integration_steps=10,
    inverse_mass_matrix=jnp.ones(2),
)
sampler3 = geomjax.lmc(
    logdensity_fn,
    step_size=1e-2,
    num_integration_steps=10,
    metric_fn=lambda x: jnp.eye(2),
)
sampler4 = geomjax.lmcmonge(
    logdensity_fn,
    step_size=1e-2,
    inverse_mass_matrix=jnp.ones(2),
    num_integration_steps=10,
    alpha2=0.0,
)

state = sampler1.init(jnp.zeros(2))
state_lmc = sampler3.init(jnp.zeros(2))
state_monge = sampler4.init(jnp.zeros(2))

state1, _ = sampler1.step(rng_key, state)
state2, _ = sampler2.step(rng_key, state)
state3, _ = sampler3.step(rng_key, state_lmc)
state4, _ = sampler4.step(rng_key, state_monge)

assert jnp.allclose(state1.position, state2.position, rtol=1e-4)
assert jnp.allclose(state2.position, state3.position, rtol=1e-4)
# Monge does not match
# assert jnp.allclose(state2.position, state4.position, rtol=1e-4)


sampler1 = geomjax.nutsrmhmc(
    logdensity_fn, step_size=1e-2, metric_fn=lambda x: jnp.eye(2)
)
sampler2 = geomjax.nuts(logdensity_fn, step_size=1e-2, inverse_mass_matrix=jnp.ones(2))
sampler3 = geomjax.nutslmc(
    logdensity_fn, step_size=1e-2, metric_fn=lambda x: jnp.eye(2)
)
sampler4 = geomjax.nutslmcmonge(
    logdensity_fn, step_size=1e-2, inverse_mass_matrix=jnp.ones(2), alpha2=0.0
)
state1, _ = sampler1.step(rng_key, state)
state2, _ = sampler2.step(rng_key, state)
state3, _ = sampler3.step(rng_key, state_lmc)
state4, _ = sampler4.step(rng_key, state_monge)
assert jnp.allclose(state1.position, state2.position, rtol=1e-4)
assert jnp.allclose(state2.position, state3.position, rtol=1e-4)
# Monge does not match
# assert jnp.allclose(state2.position, state4.position, rtol=1e-4)


print("Passed test!")
