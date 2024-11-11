# ARMA(1,1) model
# https://juanitorduz.github.io/arma_numpyro/

# %%
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class IMA11:
    def __init__(self, mu, theta, sigma_nu):
        """
        The model is:
        y_t = mu + nu_t + theta * nu_{t-1}
        """
        self.mu = mu
        self.theta = theta
        self.sigma_nu = sigma_nu

    def generate_data(self, T, seed):
        """
        Generate data from the ARMA(1,1) model.
        Important here is to use scan to generate the data.
        This is more efficient than a simple for loop.
        See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan
        """
        rng_key = jax.random.PRNGKey(seed)
        # noise
        nu = jax.random.normal(rng_key, shape=(T,)) * self.sigma_nu

        def arma_step(carry, nu):
            y_prev, nu_prev = carry
            y_t = y_prev + self.mu + self.theta * nu_prev + nu
            to_carry = (y_t, nu_prev)
            return to_carry, y_t

        initial_carry = (0.0, nu[0])
        _, y = jax.lax.scan(arma_step, initial_carry, nu)
        return y

    def get_prior(self):
        mu_prior = dist.Normal(0.0, 1.0)
        theta_prior = dist.Normal(0.0, 1.0)
        sigma_nu_prior = dist.Uniform(0.0, 10.0)
        return mu_prior, theta_prior, sigma_nu_prior

    def numpyro_model(self, y, t_future: int = 0):
        # t_future is the number of future time steps to forecast
        T = len(y)
        mu_prior, theta_prior, sigma_nu_prior = self.get_prior()
        mu = numpyro.sample("mu", mu_prior)
        theta = numpyro.sample("theta", theta_prior)
        sigma_nu = numpyro.sample("sigma_nu", sigma_nu_prior)

        def transition_fn(carry, t):
            """
            This function computes the prediction given the current model parameters and state
            and returns the prediction error.
            """
            y_prev, nu_prev = carry
            prediction = y_prev + mu + theta * nu_prev
            prediction_error = y[t] - prediction
            return (y[t], prediction_error), prediction_error

        initial_error = y[0] - mu  # initial estimate of the noise
        # key idea here is that nu_t ~ N(0, sigma_nu)
        # so that nu_t ~ N(y_t - y_{t-1} - mu - theta * nu_{t-1}, sigma_nu)
        n_timesteps = T + t_future
        # scan over all time steps
        _, errors = jax.lax.scan(
            transition_fn, (0.0, initial_error), jnp.arange(1, n_timesteps)
        )
        errors = jnp.concatenate([initial_error[None], errors])
        numpyro.sample("errors", dist.Normal(0, sigma_nu), obs=errors)

        # forecast part
        if t_future > 0:

            def prediction_fn(carry, _):
                y_prev, nu_prev = carry
                prediction = y_prev + mu + theta * nu_prev
                return (prediction, 0.0), prediction

            _, y_forecast = jax.lax.scan(
                prediction_fn, (y[-1], errors[-1]), jnp.arange(t_future)
            )
            numpyro.sample("y_forecast", dist.Normal(y_forecast, sigma_nu))

    def run_inference(
        self, x, rng_key, num_warmup=1000, num_samples=1000, num_chains=1, nuts_kwargs=None
    ):
        if nuts_kwargs is None:
            nuts_kwargs = {}
        sampler = numpyro.infer.NUTS(self.numpyro_model, **nuts_kwargs)
        mcmc = numpyro.infer.MCMC(
            sampler,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(rng_key, x)
        return mcmc

    def run_prediction(self, posterior_samples, rng_key, *model_args):
        predictive = numpyro.infer.Predictive(
            model=self.numpyro_model,
            posterior_samples=posterior_samples,
            return_sites=["y_forecast", "errors"],
        )
        return predictive(rng_key, *model_args)


# %%
import matplotlib.pyplot as plt
import arviz as az

T = 100
mu = -0.75
theta = 0.5
sigma_nu = 1.0
n_train = 80
n_test = T - n_train
seed = 42
model = IMA11(mu=mu, theta=theta, sigma_nu=sigma_nu)
true_data = model.generate_data(T=T, seed=seed)
y_train = true_data[:n_train]
y_test = true_data[n_train:]

# %%
mcmc = model.run_inference(y_train, jax.random.PRNGKey(seed), num_chains=2)

# %%
mu_prior, theta_prior, sigma_nu_prior = model.get_prior()
prior_samples = {
    "mu": mu_prior.sample(jax.random.PRNGKey(seed)),
    "theta": theta_prior.sample(jax.random.PRNGKey(seed)),
    "sigma_nu": sigma_nu_prior.sample(jax.random.PRNGKey(seed)),
}
idata = az.from_numpyro(posterior=mcmc, prior=prior_samples)
axes = az.plot_trace(
    data=idata,
    compact=True,
    lines=[("mu", {}, mu), ("theta", {}, theta), ("sigma_nu", {}, sigma_nu)],
)
plt.show()

# %%
# predictions
rng_key = jax.random.PRNGKey(seed)
posterior_samples = mcmc.get_samples()
forecast = model.run_prediction(posterior_samples, rng_key, y_train, n_test)

# %%
fig, ax = plt.subplots()
ax.plot(true_data, label="true", color = "black")
for y_forecast in forecast["y_forecast"][0:100]:
    y = jnp.concatenate([y_train, y_forecast])
    ax.plot(y, color="C0", alpha=0.10)
plt.show()

# %%
# variational inference
optimizer = numpyro.optim.Adam(step_size=5e-4)
