import pymc3 as pm

from datalib import load, plot
import util

df = load.requests_by_region_by_date()
data_plot = plot.observations_per_region(df, log_scale=True)
prob = load.construct_problem(df)
x = prob.dates
y = prob.request_counts_log10

with pm.Model() as linear:
    a = pm.Normal("a", mu=4, sigma=1)
    b = pm.Normal("b", mu=0, sigma=0.1)
    noise = pm.HalfNormal("noise", sigma=0.1)
    trend = pm.Deterministic("trend", a + b * x)
    pm.Normal("y", mu=trend, sigma=noise, observed=y)

prior_pred_samples = pm.sample_prior_predictive(samples=50, model=linear)

prior_pred_fig, prior_pred_ax = util.plot_samples(df["Date"], prior_pred_samples)
plot.observations_per_region(df, log_scale=True, fig=prior_pred_fig, ax=prior_pred_ax)
prior_pred_ax.set_title("Prior predictive check")

with linear:
    trace = pm.sample(
        2000, tune=2000, chains=2, target_accept=0.95, return_inferencedata=True
    )

post_pred_samples = pm.sample_posterior_predictive(trace, model=linear)
post_pred_fig, post_pred_ax = util.plot_pred_distribution(df["Date"], post_pred_samples)
plot.observations_per_region(df, log_scale=True, fig=post_pred_fig, ax=post_pred_ax)
post_pred_fig.show()
