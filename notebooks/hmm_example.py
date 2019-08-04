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
state_ids = np.arange(states.shape[0])


# %%
def transition_probability(x, y):
    return expon.pdf(np.sum(np.abs(states[x, :] - states[y, :])), scale=10)


# %%
cov = np.eye(2)
def emission_probability(z, x):
    return multivariate_normal.pdf(z, mean=states[x, :], cov=cov)


# %%
def initial_probability(x):
    return 1/100


# %%
hmm = HiddenMarkovModel(transition_probability, emission_probability, initial_probability, 100)

# %%
states;

# %%
true_path = list()
observations = list()
P = hmm.P

T = 10
state = np.random.choice(state_ids)
observation = multivariate_normal.rvs(mean=states[state, :], cov=cov)
true_path.append(state)
observations.append(observation)
for t in range(T-1):
    state = np.random.choice(state_ids, p=P[state, :])
    observation = multivariate_normal.rvs(mean=states[state, :], cov=cov)
    true_path.append(state)
    observations.append(observation)

# %%
viterbi_path = hmm.viterbi(observations)

# %%
viterbi_path

# %%
viterbi_states = [states[i, :] for i in viterbi_path.astype(int)]

# %%
viterbi_states

# %%
true_path

# %%
true_states = [states[i, :] for i in np.array(true_path).astype(int)]

# %%
true_states

# %%
transition_probability(55, 66)

# %%
observations

# %%
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots()
plt.xticks(np.arange(-2, 12+1, 1.0))
plt.yticks(np.arange(-2, 12+1, 1.0))
plt.grid(which="both")
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)
ax.scatter(np.vstack(viterbi_states)[:, 0], np.vstack(viterbi_states)[:, 1], marker="x")
ax.scatter(np.vstack(true_states)[:, 0], np.vstack(true_states)[:, 1], marker="+")
ax.scatter(np.vstack(observations)[:, 0], np.vstack(observations)[:, 1], marker="p")

# %%
from IPython.display import clear_output
from time import sleep
for i in range(T):
    fig, ax = plt.subplots()
    plt.xticks(np.arange(-2, 12+1, 1.0))
    plt.yticks(np.arange(-2, 12+1, 1.0))
    plt.grid(which="both")
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.scatter(
        np.vstack(viterbi_states)[i, 0],
        np.vstack(viterbi_states)[i, 1],
        marker="x",
        label="Predicted state"
    )
    ax.scatter(np.vstack(true_states)[i, 0],
               np.vstack(true_states)[i, 1],
               marker="+",
               label="True state"
    )
    ax.scatter(np.vstack(observations)[i, 0],
               np.vstack(observations)[i, 1],
               marker="p",
               label="Observation"
    )
    ax.plot(
        np.vstack(viterbi_states)[:i+1, 0],
        np.vstack(viterbi_states)[:i+1, 1],
        marker="x"
    )
    ax.plot(np.vstack(true_states)[:i+1, 0],
               np.vstack(true_states)[:i+1, 1],
               marker="+"
    )
    ax.plot(np.vstack(observations)[:i+1, 0],
               np.vstack(observations)[:i+1, 1],
               marker="p"
    )
    ax.legend()
    clear_output(wait=True)
    plt.show()
    sleep(1)
