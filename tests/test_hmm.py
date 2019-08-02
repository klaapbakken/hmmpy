import pytest
import numpy as np

from scipy.stats import norm

from hmmpy.hmm import (
    TransitionProbability,
    EmissionProbability,
    InitialProbability,
    HiddenMarkovModel,
)

TRANSITION_MATRIX = np.ones((3, 3)) * 1 / 3


@pytest.fixture
def transition_probability():
    def func(x, y):
        P = TRANSITION_MATRIX
        return P[x, y]

    transition_probability = TransitionProbability(func, 3)
    return transition_probability


@pytest.fixture
def emission_probability():
    def func(z, x):
        return norm.pdf(z, loc=x)

    emission_probability = EmissionProbability(func, 3)
    return emission_probability


@pytest.fixture
def initial_probability() -> float:
    def func(x):
        return 1 / 3

    initial_probability = InitialProbability(func, 3)
    return initial_probability


@pytest.fixture
def hidden_markov_model():
    def func1(x, y):
        P = np.ones((3, 3)) * 1 / 3
        return P[x, y]

    def func2(z, x):
        return norm.pdf(z, loc=x)

    def func3(x):
        return 1 / 3

    return HiddenMarkovModel(func1, func2, func3, 3)


class TestTransitionProbability:
    def test_eval(self, transition_probability):
        a = np.array([1, 2, 0])
        b = np.array([2, 0, 1])
        res = transition_probability.eval(a, b)
        assert res.shape == a.shape == b.shape
        assert np.sum(res) == 1


class TestEmissionProbability:
    def test_eval(self, emission_probability):
        a = np.array([1, 3, 0])
        b = np.array([2, 2, 1])
        res = emission_probability.eval(a, b)
        assert res.shape == a.shape == b.shape
        assert np.sum(res) == norm.pdf(1) * 3


class TestInitialProbability:
    def test_eval(self, initial_probability):
        a = np.array([2, 2, 1])
        res = initial_probability.eval(a)
        assert res.shape == a.shape
        assert np.sum(res) == 1


class TestHiddenMarkovModel:
    def test_object_creation(self, hidden_markov_model, transition_probability):
        assert hidden_markov_model.M == 3
        assert np.all(hidden_markov_model.P == TRANSITION_MATRIX)

    def test_viterbi(self, hidden_markov_model):
        observations = np.array([1, 2, 0])
        most_likely_path = np.array([1, 2, 0])
        viterbi_path = hidden_markov_model.viterbi(observations)
        assert np.all(viterbi_path == most_likely_path)
