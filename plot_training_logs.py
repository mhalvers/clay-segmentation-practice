# %%
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

# %%
df = pd.read_csv('logs/lightning_logs/version_11/metrics.csv')
df

# %%
df["epoch"].plot()

# %%
df["step"].plot()

# %% group by epoch and take mean
epoch_logs = df.groupby("epoch").mean()

# %% plot, choosing colors from tab10 colormap
colors = mpl.colormaps["tab10"].colors
epoch_logs.dropna(axis="columns", how="all").drop(columns=["step"]).plot(color=colors)

# %% count number of entries per epoch
df.groupby("epoch").count()["step"].plot()

# %%
epoch_logs = epoch_logs.dropna(axis="columns", how="all")
epoch_logs = epoch_logs.drop(columns=["step"])

# %% drop columsn with "step" in the name
has_step = [col for col in epoch_logs.columns if "step" in col]
epoch_logs = epoch_logs.drop(columns=has_step)


# %%
epoch_logs.plot(color=colors)
# %%
