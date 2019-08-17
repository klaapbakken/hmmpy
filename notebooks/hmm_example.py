# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: hmmpy
#     language: python
#     name: hmmpy
# ---

# %%
import numpy as np
import pandas as pd

# %%
from hmmpy.hmm import HiddenMarkovModel

# %%
from scipy.stats import multivariate_normal
from scipy.stats import expon

# %%
xs = np.repeat(np.arange(10), 10)
ys = np.tile(np.arange(10), 10)

# %%
states = np.array(list(zip(xs, ys)))


# %%
def transition_probability(x, y):
    return expon.pdf(np.sum(np.abs(x - y)), scale=10)


# %%
cov = np.eye(2)
def emission_probability(z, x):
    return multivariate_normal.pdf(z, mean=x, cov=cov)


# %%
def initial_probability(x):
    return 1/100


# %%
hmm = HiddenMarkovModel(transition_probability, emission_probability, initial_probability, states)

# %%
true_path = list()
observations = list()
P = hmm.P
state_ids = np.arange(len(states))

T = 10
state = np.random.choice(state_ids)
observation = multivariate_normal.rvs(mean=states[state, :], cov=cov)
true_path.append(states[state])
observations.append(observation)
for t in range(T-1):
    state = np.random.choice(state_ids, p=P[state, :])
    observation = multivariate_normal.rvs(mean=states[state, :], cov=cov)
    true_path.append(states[state])
    observations.append(observation)

# %%
most_likely_states = hmm.most_likely_path(observations)

# %%
from pprint import pprint as pp

# %%
pp(most_likely_states)

# %%
pp(true_path)

# %%
pp(observations)

# %%
df = pd.DataFrame({"true_states" : true_path, "predicted_states" : most_likely_states, "observation" : observations})

# %%
pp(df)

# %%
from bokeh.io import output_notebook, show

# %%
from bokeh.plotting import figure

# %%
plot = figure()

plot.circle(np.vstack(true_path)[:, 0], np.vstack(true_path)[:, 1], color="blue", size=10);
plot.diamond(np.vstack(most_likely_states)[:, 0], np.vstack(most_likely_states)[:, 1], color="red", size=10);
plot.x(np.vstack(observations)[:, 0], np.vstack(observations)[:, 1], color="green", size=10);

output_notebook()
show(plot)

# %%
p = hmm.forward_algorithm(observations)

# %%
hmm.backward_algorithm(observations)

# %%
hmm.calculate_ksi(observations)

# %%
hmm.calculate_gamma()
