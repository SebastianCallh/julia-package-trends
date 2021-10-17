# %%
import pymc3
from sklearn.preprocessing import StandardScaler

from datalib import load, plot

df = load.requests_by_region_by_date()
plt = plot.observations_per_region(df, log_scale=True)
# %%
scaler = StandardScaler()
x = scaler.fit_transform(df["Date"].values.reshape(-1, 1)).flatten()
y = df["Request count (log10)"]

# %%
