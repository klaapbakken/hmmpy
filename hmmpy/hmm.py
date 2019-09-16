from typing import Callable, Any
from functools import reduce, partial

import numpy as np
from numpy import ma

from math import exp, sqrt, pi

from itertools import product

class InitialProbability:
    """Class for representing and evaluating initial probabilties."""

    def __init__(self, initial_probability: Callable[[Any], float], states: list):
        self.states = states
        self.n = len(states)
        self.pi: Callable[[Any], float] = initial_probability
        self.max_index: int = self.n - 1

    def eval(self, x: np.ndarray):
        assert np.max(x) <= self.max_index
        return np.array(list(map(lambda x: self.pi(self.states[x]), x)))

class TransitionProbability:
    """Class for representing and evaluating transition probabilties."""

    def __init__(
        self, transition_probability: Callable[[Any, Any], float], states: list
    ):
        self.states = states
        self.n = len(states)
        self.p: Callable[[Any, Any], float] = transition_probability
        self.max_index: int = self.n - 1

    def eval(self, x: np.ndarray, y: np.ndarray):
        assert np.max(x) <= self.max_index and np.max(y) <= self.max_index
        return np.array(list(map(lambda x: self.p(self.states[x[0]], self.states[x[1]]), zip(x, y))))

class EmissionProbability:
    """Class for representing and evaluating emission probabilties."""

    def __init__(self, emission_probability: Callable[[Any, Any], float], states: list):
        self.n = len(states)
        self.states = states
        self.l: Callable[[Any, Any], float] = emission_probability
        self.max_index: int = self.n - 1

    def eval(self, z: list, x: np.ndarray):
        assert np.max(x) <= self.max_index
        return np.array([self.l(obs, self.states[state]) for obs, state in zip(z, x)])

class DiscreteEmissionProbability():
    def __init__(self, emission_probability, states, symbols):
        self.symbol_id_dictionary = {k : v for k, v in zip(symbols, range(len(symbols)))}
        self.K = len(symbols)
        self.M = len(states)
        self.l = emission_probability
        self.b = np.array(list(map(lambda x: self.l(x[0], x[1]), product(symbols, states)))).reshape(self.K, self.M)
        self.b = self.b / np.sum(self.b, axis=0)

    def eval(self, z: list, x: np.ndarray):
        return np.array([self.b[self.symbol_id_dictionary[a], b] for a, b in zip(z, x)])

class GaussianEmissionProbability():
    def __init__(self, mus: list, sigmas: list):
        def emission_probability(z, x):
            return 1/(sqrt(2*pi)*sigmas[x])*exp((x - mus[x])**2/(2*sigmas[x]**2))
        self.l = emission_probability

    def eval(self, z: np.ndarray, x: np.ndarray):
        #Can do apply along axis.
        return np.array([self.l(z, x) for z, x in zip(z, x)])

class HiddenMarkovModel:
    """Class that implements functionality related to Hidden Markov Models."""

    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        emission_probability: Callable[[Any, int], float],
        initial_probability: Callable[[int], float],
        states: list,
    ):
        self.states = states
        self.M: int = len(states)
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability, self.states
        )
        self.emission_probability: EmissionProbability = EmissionProbability(
            emission_probability, self.states
        )
        self.initial_probability: InitialProbability = InitialProbability(
            initial_probability, self.states
        )
        self.alpha = None
        self.beta = None
        self.ksi = None
        self.gamma = None

        states_repeated: np.ndarray = np.repeat(self.state_ids, self.M)
        states_tiled: np.ndarray = np.tile(self.state_ids, self.M)

        self.P: np.ndarray = self.transition_probability.eval(
            states_repeated, states_tiled
        ).reshape(self.M, self.M)

        sumP: np.ndarray = np.sum(self.P, axis=1)
        self.P = (self.P.T * 1 / sumP).T

    def naive_viterbi(self, z: list):
        N: int = len(z)
        delta: np.ndarray = np.zeros((N, self.M))
        phi: np.ndarray = np.zeros((N, self.M))

        delta[0, :] = self.initial_probability.eval(self.state_ids) * (
            self.emission_probability.eval([z[0]] * self.M, self.state_ids)
        )

        phi[0, :] = 0

        for n in np.arange(1, N):
            # Multiply delta by each column in P
            # In resulting matrix, for each column, find max entry
            l: np.ndarray = self.emission_probability.eval(
                [z[n]] * self.M, self.state_ids
            )
            delta[n, :] = l * np.max((delta[n - 1, :] * self.P.T).T, axis=0)

            phi[n, :] = np.argmax((delta[n - 1, :] * self.P.T).T, axis=0)

        x_star: np.ndarray = np.zeros((N,))
        x_star[N - 1] = np.argmax(delta[N - 1, :])

        for n in np.arange(N - 2, -1, -1):
            x_star[n] = phi[n + 1, x_star[n + 1].astype(int)]

        return x_star.astype(int)
    
    def viterbi(self, z: list):
        N: int = len(z)
        delta: np.ndarray = np.zeros((N, self.M))
        phi: np.ndarray = np.zeros((N, self.M))
        log_P = ma.log(self.P).filled(-np.inf)
        

        delta[0, :] = np.log(self.initial_probability.eval(self.state_ids)) + np.log(self.emission_probability.eval([z[0]] * self.M, self.state_ids))
        phi[0, :] = 0

        for n in np.arange(1, N):
            # Multiply delta by each column in P
            # In resulting matrix, for each column, find max entry
            log_l: np.ndarray = np.log(self.emission_probability.eval(
                [z[n]] * self.M, self.state_ids
            ))
            delta[n, :] = log_l + np.max((np.expand_dims(delta[n - 1, :], axis=1) + log_P), axis=0)
            phi[n, :] = np.argmax((np.expand_dims(delta[n - 1, :], axis=1) + log_P), axis=0)

        q_star = np.zeros((N, ))
        q_star[N - 1] = np.argmax(delta[N - 1, :])

        for n in np.arange(N - 2, -1, -1):
            q_star[n] = phi[n + 1, q_star[n + 1].astype(int)]

        return q_star.astype(int)

    def decode(self, z: list):
        state_ids = self.viterbi(z)
        return list(map(lambda x: self.states[x], state_ids))

    def forward_algorithm(self, z: list):
        N: int = len(z)

        alpha = np.zeros((N, self.M))
        c = np.zeros((N,))

        alpha[0, :] = self.emission_probability.eval(
            [z[0]] * self.M, self.state_ids
        ) * self.initial_probability.eval(self.state_ids)
        c[0] = np.reciprocal(np.sum(alpha[0, :]))
        alpha[0, :] = alpha[0, :]*c[0]

        for n in np.arange(N - 1):
            alpha[n + 1, :] = np.sum(alpha[n, :] * self.P, axis=1) * (
                self.emission_probability.eval([z[n + 1]] * self.M, self.state_ids)
            )
            c[n+1] = np.reciprocal(np.sum(alpha[n+1, :]))
            alpha[n + 1, :] = alpha[n + 1, :] * c[n+1]

        self.c = c
        self.alpha = alpha

    def backward_algorithm(self, z: list):
        N = len(z)
        beta = np.zeros((N, self.M))
        beta[N - 1, :] = 1

        for n in np.arange(N - 2, -1, -1):
            b = self.emission_probability.eval(
                np.array([z[n + 1]] * self.M), np.arange(self.M)
            )

            beta[n, :] = np.sum(self.P * b * beta[n + 1, :], axis=1)

        self.beta = beta * self.c[:, np.newaxis]

    def calculate_ksi(self, z: list):
        N = len(z)

        ksi = np.zeros((N - 1, self.M, self.M))
        for n in range(N - 1):
            b = self.emission_probability.eval(
                     np.array([z[n + 1]] * self.M), self.state_ids
                     )
            ksi[n, :, :] = (self.P * b * self.beta[n + 1, :]) * self.alpha[n, :][:, np.newaxis]
            ksi[n, :, :] = ksi[n, :, :]/np.sum(ksi[n, :, :])

        self.ksi = ksi

    def calculate_gamma(self):
        alpha_beta_product = self.alpha * self.beta
        sum_over_all_states = np.sum(alpha_beta_product, axis=1)
        self.gamma = alpha_beta_product / sum_over_all_states[:, np.newaxis]

    def observation_log_probability(self, z):
        self.forward_algorithm(z)
        return -np.sum(self.c)



class DiscreteHiddenMarkovModel(HiddenMarkovModel):
    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        emission_probability: Callable[[Any, int], float],
        initial_probability: Callable[[int], float],
        states: list,
        symbols: list
    ):
        self.states = states
        self.symbols = symbols
        self.M: int = len(states)
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability, self.states
        )
        self.emission_probability: DiscreteEmissionProbability = DiscreteEmissionProbability(
            emission_probability, self.states, self.symbols
        )
        self.initial_probability: InitialProbability = InitialProbability(
            initial_probability, self.states
        )
        self.alpha = None
        self.beta = None
        self.ksi = None
        self.gamma = None

        states_repeated: np.ndarray = np.repeat(self.state_ids, self.M)
        states_tiled: np.ndarray = np.tile(self.state_ids, self.M)

        self.P: np.ndarray = self.transition_probability.eval(
            states_repeated, states_tiled
        ).reshape(self.M, self.M)

        sumP: np.ndarray = np.sum(self.P, axis=1)
        self.P = (self.P.T * 1 / sumP).T

    def learn(self, z, symbols):
        K = len(symbols)
        a_u = np.sum(self.ksi, axis=0) 
        a_l = np.sum(self.gamma, axis=0)

        b_u = np.ones((K, self.M))
        for k, o in enumerate(symbols):
            ts = np.where(np.array(z) == o)
            partial_gamma = np.zeros(self.gamma.shape)
            partial_gamma[ts, :] = self.gamma[ts, :]
            b_u[k, :] = np.sum(partial_gamma, axis=0)
        b_l = np.sum(self.gamma, axis=0)

        pi = self.gamma[0, :]

        return a_u, a_l, b_u, b_l, pi

    def learn_from_sequence(self, zs, symbols):
        E = len(zs)
        a_us = []
        a_ls = []
        b_us = []
        b_ls = []
        pis = []
        for z in zs:
            self.forward_algorithm(z)
            self.backward_algorithm(z)
            self.calculate_gamma()
            self.calculate_ksi(z)
            a_u, a_l, b_u, b_l, pi = self.learn(z, symbols)
            a_us.append(a_u)
            a_ls.append(a_l)
            b_us.append(b_u)
            b_ls.append(b_l)
            pis.append(pi)
        
        self.P = sum(a_us)/sum(a_ls)
        print(self.P.shape)
        self.b = sum(b_us)/sum(b_ls)[:, np.newaxis]
        print(self.b.shape)
        self.pi = sum(pis)/E
        print(self.pi.shape)
        return

        assert self.b.shape == (self.emission_probability.K, self.emission_probability.M)
        self.emission_probability.b = self.b

    def reestimate(self, zs: list, iterations=10):
        for _ in range(iterations):
            self.learn_from_sequence(zs)

class GaussianHiddenMarkovModel(HiddenMarkovModel):
    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        initial_probability: Callable[[int], float],
        states: list,
        mus: list,
        sigmas: list
    ):
        self.states = states
        self.M: int = len(states)
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability, self.states
        )
        self.emission_probability: GaussianEmissionProbability = GaussianEmissionProbability(
            mus, sigmas
        )
        self.initial_probability: InitialProbability = InitialProbability(
            initial_probability, self.states
        )
        self.alpha = None
        self.beta = None
        self.ksi = None
        self.gamma = None

        states_repeated: np.ndarray = np.repeat(self.state_ids, self.M)
        states_tiled: np.ndarray = np.tile(self.state_ids, self.M)

        self.P: np.ndarray = self.transition_probability.eval(
            states_repeated, states_tiled
        ).reshape(self.M, self.M)

        sumP: np.ndarray = np.sum(self.P, axis=1)
        self.P = (self.P.T * 1 / sumP).T

    def learn(self, z, mu, gamma, ksi):
        a_u = np.sum(self.ksi, axis=0) 
        a_l = self.gamma[:-1, np.newaxis]

        sigma_u = np.sum(gamma, axis=0)*((z - mu)*(z - mu).T)
        sigma_l = np.sum(self.gamma, axis=0)

        pi = self.gamma[0, :]

        return a_u, a_l, sigma_u, sigma_l, pi
    
    def learn_mu(self, z):
        mu_u = np.sum(self.gamma*np.array(z)[:, np.newaxis], axis=0)
        mu_l = np.sum(self.gamma, axis=0)

        return mu_u, mu_l

    def learn_from_sequence(self, zs):
        E = len(zs)
        mu_us = []
        mu_ls = []

        gammas = []
        ksis = []

        for z in zs:
            self.forward_algorithm(z)
            self.backward_algorithm(z)
            self.calculate_gamma()
            self.calculate_ksi(z)

            gammas.append(self.gamma)
            ksis.append(self.ksi)

            mu_u, mu_l = self.learn_mu(z)
            mu_us.append(mu_u)
            mu_ls.append(mu_l)

        self.mu = sum(mu_us)/sum(mu_ls)[:, np.newaxis]

        a_us = []
        a_ls = []
        sigma_us = []
        sigma_ls = []
        pis = []
        for z, gamma, ksi in zip(zs, gammas, ksis):
            a_u, a_l, sigma_u, sigma_l, pi = self.learn(z, self.mu, gamma, ksi)
            a_us.append(a_u)
            a_ls.append(a_l)
            sigma_us.append(sigma_u)
            sigma_ls.append(sigma_l)
            pis.append(pi)
        
        self.P = sum(a_us)/sum(a_ls)[:, np.newaxis]
        self.sigma = sum(sigma_us)/sum(sigma_ls)[:, np.newaxis]
        self.pi = sum(pis)/E
        self.emission_probability = GaussianEmissionProbability(self.mu, self.sigma)

    def reestimate(self, zs: list, iterations=10):
        for _ in range(iterations):
            self.learn_from_sequence(zs)
        

