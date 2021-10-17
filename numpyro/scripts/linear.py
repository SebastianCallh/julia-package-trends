#!/usr/bin/env python3
from typing import Dict
import pandas as pd
from matplotlib import pyplot as plt

from jax import random
from jax import numpy as np
import numpyro as npo
import numpyro.distributions as dist
from numpyro.infer import Predictive, MCMC, NUTS

from datalib import load, plot, params

df = load.requests_by_region_by_date()
prob = load.construct_problem(df)
prior_params = params.priors()
rng_key = random.PRNGKey(0)

def linear(x, params: Dict[str, float], y=None):
    N = x.shape[0]
    a = npo.sample('a', dist.Normal(params["alpha"]["loc"], params["alpha"]["scale"]))
    b = npo.sample('b', dist.Normal(params["beta"]["loc"], params["beta"]["scale"]))
    sigma = npo.sample("sigma", dist.HalfNormal(params["sigma"]["scale"]))
    with npo.plate('N', N):
        npo.sample('obs', dist.Normal(a + b*x, sigma), obs=y)

xx = np.linspace(prob.dates.min(), prob.dates.max(), num=250)
xx_unnormalized = pd.to_datetime(prob.feature_scaler.inverse_transform(xx.reshape(1, -1)).flatten())
rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(linear, num_samples=100)
prior_preds = prior_predictive(rng_key_, x=xx, params=prior_params["linear"])

yy = prior_preds["a"].reshape(-1, 1) + prior_preds["b"].reshape(-1, 1) @ xx.reshape(1, -1)
fig, ax = plt.subplots()
ax.plot(
    xx_unnormalized,
    yy.T,
    color="blue",
    alpha=0.2
)
plot.observations_per_region(df, log_scale=True, fig=fig, ax=ax)
fig.show()

rng_key, rng_key_ = random.split(rng_key)
mcmc = MCMC(NUTS(linear), num_warmup=1000, num_samples=2000)
mcmc.run(
    rng_key_,
    x=prob.dates,
    y=prob.request_counts_log10,
    params=prior_params["linear"]
)
mcmc.print_summary()

rng_key, rng_key_ = random.split(rng_key)
posterior_predictrive = Predictive(linear, mcmc.get_samples())
posterior_preds = posterior_predictrive(rng_key_, x=xx, params=prior_params["linear"])
post_mean = posterior_preds["obs"].mean(0)
post_std = posterior_preds["obs"].std(0)
fig, ax = plt.subplots()
ax.plot(
    xx_unnormalized,
    post_mean,
    color="blue",
    alpha=0.2,
    label="Posterior mean"
)
ax.fill_between(
    xx_unnormalized,
    post_mean + 2*post_std,
    post_mean - 2*post_std,
    alpha=0.4,
    label="95% credible bands"
)
plot.observations_per_region(df, log_scale=True, fig=fig, ax=ax)
fig.show()
