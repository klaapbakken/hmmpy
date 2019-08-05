import pytest
import numpy as np

from scipy.stats import norm

from hmmpy.hmm import (
    TransitionProbability,
    EmissionProbability,
    InitialProbability,
    HiddenMarkovModel,
)

np.random.seed(0)

OBSERVATIONS = np.random.choice(np.arange(10), size=10).tolist()
TRANSITION_MATRIX = np.ones((10, 10)) * 1 / 10
STATES = np.arange(10).tolist()


@pytest.fixture
def transition_probability():
    def func(x, y):
        P = TRANSITION_MATRIX
        return P[x, y]

    transition_probability = TransitionProbability(func, STATES)
    return transition_probability


@pytest.fixture
def emission_probability():
    def func(z, x):
        return norm.pdf(z, loc=x)

    emission_probability = EmissionProbability(func, STATES)
    return emission_probability


@pytest.fixture
def initial_probability() -> float:
    def func(x):
        return 1 / 10

    initial_probability = InitialProbability(func, STATES)
    return initial_probability


@pytest.fixture
def hidden_markov_model():
    def func1(x, y):
        P = TRANSITION_MATRIX
        return P[x, y]

    def func2(z, x):
        return norm.pdf(z, loc=x)

    def func3(x):
        return 1 / 10

    return HiddenMarkovModel(func1, func2, func3, STATES)


class TestTransitionProbability:
    def test_eval(self, transition_probability):
        a = np.array([1, 2, 0])
        b = np.array([2, 0, 1])
        res = transition_probability.eval(a, b)
        assert res.shape == a.shape == b.shape
        assert np.sum(res) == pytest.approx(3 / 10)


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
        assert np.sum(res) == pytest.approx(3 / 10)


class TestHiddenMarkovModel:
    def test_object_creation(self, hidden_markov_model, transition_probability):
        assert hidden_markov_model.M == 10
        assert np.all(hidden_markov_model.P == TRANSITION_MATRIX)

    def test_viterbi(self, hidden_markov_model):
        most_likely_path = OBSERVATIONS
        viterbi_path = hidden_markov_model.viterbi(OBSERVATIONS)
        assert np.all(viterbi_path == np.array(most_likely_path))

    def test_most_proable_path(self, hidden_markov_model):
        most_likely_states = hidden_markov_model.most_likely_path(OBSERVATIONS)
        assert most_likely_states == list(map(lambda x: x, OBSERVATIONS))

    def test_forward_algorithm(self, hidden_markov_model):
        observations = np.random.choice(np.arange(10), size=10)
        probability = hidden_markov_model.forward_algorithm(observations)
        assert 0 < probability <= 1
