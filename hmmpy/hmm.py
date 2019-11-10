import numpy as np
from numpy import ma
from numpy.linalg import det

from scipy.stats import multivariate_normal

from math import exp
from math import sqrt
from math import pi

from itertools import product
from functools import reduce
from functools import partial

from typing import Callable
from typing import Any
from typing import Tuple
from typing import List
from typing import Type
from typing import Dict

from abc import ABC, abstractmethod

import warnings


class EmissionProbabilityBase(ABC):
    @abstractmethod
    def eval_to_array(self, z, x):
        pass

    @abstractmethod
    def l(self, z):
        pass


class InitialProbability:
    """Class for representing and evaluating initial probabilties.
    
    Parameters:
    ---
    initial_probability -- A function, taking a single argument, with the argument being
    an element present in the "states" argument, that returns the probability of starting of the supplied state.
    states -- A list of all states in the state space.
    """

    def __init__(
        self,
        initial_probability: Callable[[Any], float],
        states: List[Any],
        enable_warnings: bool = False,
    ):
        self.states: List[Any] = states
        self.state_ids: np.ndarray = np.arange(self.M)
        self.pi_function: Callable[[Any], float] = initial_probability
        self.enable_warnings = enable_warnings

    def eval_to_array(
        self, pi_function: Callable[[Any], float], x: np.ndarray
    ) -> np.ndarray:
        """Get corresponding initial probability for states identified by state IDs in x. 
        State IDs is the index of the states in the list passed in the constructor. 
        """
        return np.array(list(map(lambda x: pi_function(self.states[x]), x)))

    @property
    def M(self):
        return len(self.states)

    @property
    def pi_function(self):
        return None

    @pi_function.setter
    def pi_function(self, value):
        self.pi = self.eval_to_array(value, self.state_ids)

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, value):
        unscaled_pi = value
        scaled_pi = unscaled_pi / np.sum(unscaled_pi)
        if (
            not np.all(np.isclose(unscaled_pi, scaled_pi, atol=1e-3))
            and self.enable_warnings
        ):
            warnings.warn(
                f"Unscaled initial probability vector supplied to setter: {np.sum(unscaled_pi)} != {1}",
                UserWarning,
            )
        self._pi = scaled_pi


class TransitionProbability:
    """Class for representing and evaluating transition probabilties.
    
    Parameters:
    ---
    transition_probability  -- A function, taking two arguments, with both arguments being
    actual elements of the state space, that returns the probability of moving from the first argument
    to the second argument.
    states -- A list of all states in the state space.
    """

    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        states: List[Any],
        frozen_mask=None,
        enable_warnings=False,
    ):
        self.states: List[Any] = states
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        # Positions that are zero in "frozen_mask" correspond to zero transition probabilities in the transition matrix
        self.frozen_mask: np.ndarray = frozen_mask
        self.enable_warnings = enable_warnings
        self.P_function: Callable[[Any, Any], float] = transition_probability

    def eval_to_array(
        self, P_function: Callable[[Any, Any], float], x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Returns an array of transition probabilities, with the ith element being 
        the probability of transitioning from the ith element of the first argument to
        the ith element of the second argument. The elements should be state IDs, not states. 
        """
        return np.array(
            list(
                map(
                    lambda x: P_function(self.states[x[0]], self.states[x[1]]),
                    zip(x, y),
                )
            )
        )

    @property
    def M(self):
        return len(self.states)

    @property
    def P_function(self):
        return None

    @P_function.setter
    def P_function(self, value):
        states_repeated: np.ndarray = np.repeat(self.state_ids, self.M)
        states_tiled: np.ndarray = np.tile(self.state_ids, self.M)
        self.P: np.ndarray = self.eval_to_array(
            value, states_repeated, states_tiled
        ).reshape(self.M, self.M)

    @property
    def P(self) -> np.ndarray:
        return self._P

    @P.setter
    def P(self, value: np.ndarray):
        if self.frozen_mask is not None:
            unscaled_P = value * self.frozen_mask
        else:
            unscaled_P: np.ndarray = value
        sum_unscaled_P: np.ndarray = np.sum(unscaled_P, axis=1)
        scaled_P = (unscaled_P.T / sum_unscaled_P).T
        if (
            not np.all(np.isclose(unscaled_P, scaled_P, atol=1e-3))
            and self.enable_warnings
        ):
            warnings.warn(
                f"Unscaled transition matrix supplied to setter: {np.sum(unscaled_P)} != {self.M}",
                UserWarning,
            )
        self._P = scaled_P


class EmissionProbability(EmissionProbabilityBase):
    """Class for representing and evaluating emission probabilties.
    
    Parameters:
    ---
    emission_probability -- A function, that takes an observation as its first argument and a state as its
    second argument and returns the probability of observing the observation given that we are in the supplied state.
    states -- A list of all the states in the state space.
    """

    def __init__(
        self,
        emission_probability: Callable[[Any, Any], float],
        states: List[Any],
        enable_warnings: bool = False,
    ):
        self.states: List[Any] = states
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.l_function: Callable[[Any, Any], float] = emission_probability
        self.enable_warnings = enable_warnings

    def eval_to_array(self, z: Any, x: np.ndarray) -> np.ndarray:
        """Returns an array of emission probabilities, with the ith element being 
        the probability of observing the ith element of the first argument when in
        the state identified by the state ID in the ith element of the second argument.
        """
        return np.squeeze(
            np.array(
                [
                    self.l_function(observation, self.states[state_id])
                    for observation, state_id in zip([z] * x.shape[0], x)
                ]
            )
        )

    @property
    def M(self):
        return len(self.states)

    def l(self, z: List[Any]):
        """Creates a 2-dimensional array of emission probabilities for the observations at various times, for various states.
        
        Parameters:
        ---
        z -- A list of observations of the type expected by the InitialProbability class.
        """
        N: int = len(z)
        l_array: np.ndarray = np.zeros((N, self.M))
        for n in range(N):
            l_array[n, :] = self.eval_to_array(z[n], self.state_ids)
        # This is a hacky solution to a problem that should probably be handled in different manner.
        # The underlying is issue is that zero, or close to zero, values, are propogated throughout the algorithm and leads to division by zero at later stages.
        return np.clip(l_array, a_min=1e-9, a_max=None)


class HiddenMarkovModel:
    """Class that implements functionality related to Hidden Markov Models.
    Supply functions as expected by the classes InitialProbability, EmissionProbability 
    and TransitionProbability.

    Parameters:
    ---
    transition_probability -- As in TransitionProbability.
    emission_probability -- As in EmissionProbability.
    initial_probability -- As in InitialProbability.
    states -- A list of all the states in the state space.
    enable_warnings -- Boolean. Indicates whether warnings should be displayed.
    frozen_mask -- A matrix with 0's and 1's indicating whether certain transitions should be illegal.
    Transition from state i to to state j is set to zero if (i, j) in frozen_mask is zero. 
    """

    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        emission_probability: Callable[[Any, int], float],
        initial_probability: Callable[[int], float],
        states: List[Any],
        enable_warnings: bool = False,
        frozen_mask: np.ndarray = None,
    ):
        self.states: List[Any] = states
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.enable_warnings: bool = enable_warnings
        self.frozen_mask = frozen_mask
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability,
            self.states,
            enable_warnings=self.enable_warnings,
            frozen_mask=self.frozen_mask
        )
        self.emission_probability: EmissionProbability = EmissionProbability(
            emission_probability, self.states, enable_warnings=self.enable_warnings
        )
        self.initial_probability: InitialProbability = InitialProbability(
            initial_probability, self.states, enable_warnings=self.enable_warnings
        )

    @property
    def M(self):
        return len(self.states)

    @property
    def pi(self):
        return self.initial_probability.pi

    @pi.setter
    def pi(self, value):
        self.initial_probability.pi = value

    @property
    def P(self):
        return self.transition_probability.P

    @P.setter
    def P(self, value):
        self.transition_probability.P = value

    def l(self, z):
        return self.emission_probability.l(z)

    def viterbi(self, z: List[Any]) -> np.ndarray:
        """Run Viterbi in order to obtain the state sequence that maximizes the posterior probability.
        In other words, the argmax of the probability of a state sequence conditioned the observed sequence.
        A wrapper around the internals.

        Parameters:
        ---
        z -- List of observations.
        """
        return self.log_viterbi_internals(z, self.P, self.l(z), self.pi)

    @staticmethod
    def viterbi_internals(
        z: List[Any], P: np.ndarray, l: np.ndarray, pi: np.ndarray
    ) -> np.ndarray:
        """Simple, self-contained, straight-forward implementation of Viterbi. 
        Should probably not be used, use log_viterbi_internals instead.
        
        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        pi -- The initial probabilities.
        l -- The emission probabilities.
        """
        N: int = len(z)
        assert pi.shape[0] == l.shape[1]
        M: int = pi.shape[0]

        delta: np.ndarray = np.zeros((N, M))
        phi: np.ndarray = np.zeros((N, M))

        delta[0, :] = pi * l[0, :]
        phi[0, :] = 0

        n: int
        for n in np.arange(1, N):
            # Multiply delta by each column in P
            # In resulting matrix, for each column, find max entry
            delta[n, :] = l[n, :] * np.max((delta[n - 1, :] * P.T).T, axis=0)
            phi[n, :] = np.argmax((delta[n - 1, :] * P.T).T, axis=0)

        x_star: np.ndarray = np.zeros((N,))
        x_star[N - 1] = np.argmax(delta[N - 1, :])

        for n in np.arange(N - 2, -1, -1):
            x_star[n] = phi[n + 1, x_star[n + 1].astype(int)]

        return x_star.astype(int)

    @staticmethod
    def log_viterbi_internals(
        z: List[Any], P: np.ndarray, l: np.ndarray, pi: np.ndarray
    ) -> np.ndarray:
        """Viterbi in log-space. More stable, i.e. resistant to almost-zero values, than the regular implementation.
        
        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        pi -- The initial probabilities.
        l -- The emission probabilities.
        """
        N: int = len(z)
        assert pi.shape[0] == l.shape[1]
        M = pi.shape[0]

        delta: np.ndarray = np.zeros((N, M))
        phi: np.ndarray = np.zeros((N, M))
        log_P = ma.log(P).filled(-np.inf)

        delta[0, :] = ma.log(pi).filled(-np.inf) + ma.log(l[0, :]).filled(-np.inf)
        phi[0, :] = 0

        n: int
        for n in np.arange(1, N):
            # Multiply delta by each column in P
            # In resulting matrix, for each column, find max entry
            log_l: np.ndarray = ma.log(l[n, :]).filled(-np.inf)
            delta[n, :] = log_l + np.max(
                (np.expand_dims(delta[n - 1, :], axis=1) + log_P), axis=0
            )
            phi[n, :] = np.argmax(
                (np.expand_dims(delta[n - 1, :], axis=1) + log_P), axis=0
            )

        q_star = np.zeros((N,))
        q_star[N - 1] = np.argmax(delta[N - 1, :])

        n: int
        for n in np.arange(N - 2, -1, -1):
            q_star[n] = phi[n + 1, q_star[n + 1].astype(int)]

        return q_star.astype(int)

    def decode(self, z: List[Any]) -> List[Any]:
        """The Viterbi method returns an array of state ids.
        This returns the corresponding states instead.

        Parameters:
        ---
        z -- List of observations.
        """
        state_ids: np.ndarray = self.viterbi(z)
        return list(map(lambda x: self.states[x], state_ids))

    def forward_algorithm(self, z: List[Any]):
        """Run the forward algorithm with object-specific arguments to the internals."""
        self.c, self.alpha = self.forward_algorithm_internals(
            z, self.P, self.l(z), self.pi
        )

    @staticmethod
    def forward_algorithm_internals(
        z: List[Any], P: np.ndarray, l: np.ndarray, pi: np.ndarray
    ):
        """The actual implementation of the forward-algorithm. Returns scalings and alpha. 
        Scaling is needed to avoid numerical errors for longer observation sequences. 
        See Rabiner's "Fundamentals of Speech Processing" or similar resource.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        pi -- The initial probabilities.
        l -- The emission probabilities.
        """
        N: int = len(z)
        assert pi.shape[0] == l.shape[1]
        M: int = pi.shape[0]

        alpha: np.ndarray = np.zeros((N, M))
        c: np.ndarray = np.zeros((N,))

        alpha[0, :] = l[0, :] * pi
        c[0] = np.reciprocal(np.sum(alpha[0, :]))
        alpha[0, :] = alpha[0, :] * c[0]

        n: int
        for n in np.arange(N - 1):
            alpha[n + 1, :] = np.sum(alpha[n, :][:, np.newaxis] * P, axis=0) * (
                l[n + 1, :]
            )
            c[n + 1] = np.reciprocal(np.sum(alpha[n + 1, :]))
            alpha[n + 1, :] = alpha[n + 1, :] * c[n + 1]

        return c, alpha

    def backward_algorithm(self, z: List[Any]) -> None:
        """Wrapper around the backward algorithm, calling the internals with object-specific attributes.
        
        Parameters:
        ---
        z -- List of observations.
        """
        assert hasattr(self, "c"), "Run forward algorithm first!"
        self.beta: np.ndarray = self.backward_algorithm_internals(
            z, self.P, self.l(z), self.pi, self.c
        )

    @staticmethod
    def backward_algorithm_internals(
        z: List[Any], P: np.ndarray, l: np.ndarray, pi: np.ndarray, c: np.ndarray
    ):
        """The actual implementation of the backward-algorithm. Returns beta, an array.
        See Rabiner's "Fundamentals of Speech Processing" or similar resource.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        l -- The emission probabilities.
        pi -- The initial probabilities.
        """
        N: int = len(z)
        assert pi.shape[0] == l.shape[1]
        M: int = pi.shape[0]

        beta: np.ndarray = np.zeros((N, M))
        beta[N - 1, :] = 1 * c[N - 1]

        n: int
        for n in np.arange(N - 2, -1, -1):
            b = l[n + 1, :]

            beta[n, :] = np.sum(P * b * beta[n + 1, :], axis=1) * c[n]

        return beta

    def forward_backward_algorithm(self, z: List[Any]):
        """Runs both forward and backward algorithm in correct order and computes ksi and gamma.
        See Rabiner for information on interpretation of these values. 
        
        Parameters:
        ---
        z -- List of observations.
        """
        self.forward_algorithm(z)
        self.backward_algorithm(z)

        alpha: np.ndarray = self.alpha
        beta: np.ndarray = self.beta

        self.gamma: np.ndarray = self.calculate_gamma(alpha, beta)
        self.ksi: np.ndarray = self.calculate_ksi(z, self.P, self.l(z), alpha, beta)

    @staticmethod
    def calculate_ksi(
        z: List[Any], P: np.ndarray, l: np.ndarray, alpha: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Compute ksi. The interpretation of ksi is the expected number of transitions from state i to j at time t.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        l -- The emission probabilities.
        pi -- The initial probabilities.
        """
        N: int = len(z)
        assert l.shape[0] == alpha.shape[0] == beta.shape[0] == N
        assert l.shape[1] == alpha.shape[1] == beta.shape[1]
        M: int = alpha.shape[1]

        ksi: np.ndarray = np.zeros((N - 1, M, M))
        n: int
        for n in range(N - 1):
            b: np.ndarray = l[n + 1, :]
            ksi[n, :, :] = (P * b * beta[n + 1, :]) * alpha[n, :][:, np.newaxis]
            ksi[n, :, :] = ksi[n, :, :] / np.sum(ksi[n, :, :])

        return ksi

    @staticmethod
    def calculate_gamma(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute gamma. The interpretation of gamma is the expected number of transitions away from state i at time t.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        l -- The emission probabilities.
        pi -- The initial probabilities.
        """
        alpha_beta_product: np.ndarray = alpha * beta
        sum_over_all_states: np.ndarray = np.sum(alpha_beta_product, axis=1)
        gamma: np.ndarray = alpha_beta_product / sum_over_all_states[:, np.newaxis]
        return gamma

    def baum_welch(self, zs: List[List[Any]]):
        """Baum-Welch algorithm. Expectation-maximization. Updates the parameters to increase the expected log-probability of the observation sequence(s).

        Parameters:
        ---
        zs -- A list of observation sequences. 
        """
        P_numerators: List[np.ndarray] = []
        P_denominators: List[np.ndarray] = []
        pis = []

        # Compute the log-probability for each of the observation sequences.
        E: int = len(zs)
        ksis: List[np.ndarray] = []
        gammas: List[np.ndarray] = []
        log_probs_list: List[float] = []
        z: List[Any]
        for z in zs:
            self.forward_backward_algorithm(z)
            log_prob: float = -np.sum(np.log(self.c))
            ksis.append(self.ksi)
            gammas.append(self.gamma)
            log_probs_list.append(log_prob)
        log_probs: np.ndarray = np.array(log_probs_list)

        # Scaling for each observation when summing over results from multiple observations
        min_log_prob: float = np.min(log_probs)
        revised_scalings: np.ndarray = np.exp(min_log_prob - log_probs)

        revised_scaling: int
        ksi: np.ndarray
        gamma: np.ndarray
        for gamma, ksi, revised_scaling in zip(gammas, ksis, revised_scalings):
            P_numerator: np.ndarray
            P_denominator: np.ndarray
            P_numerator, P_denominator = self.calculate_inner_transition_probability_sums(
                ksi, gamma
            )
            P_numerators.append(P_numerator * revised_scaling)
            P_denominators.append(P_denominator * revised_scaling)
            pi = gamma[0, :]
            pis.append(pi)

        self.P = sum(P_numerators) / sum(P_denominators)[:, np.newaxis]
        self.pi = sum(pis) / E
        

    @staticmethod
    def calculate_inner_transition_probability_sums(ksi: np.ndarray, gamma: np.ndarray):
        """Computing the sums required for the updated transition probabiltites.
        
        Parameters:
        ---
        ksi -- 3-dimensional array. First axis is time index, second and third are state indices.
        gamma -- 2-dimensional array. First axis is time index, second is state index. 
        """
        numerator_sum: np.ndarray = np.sum(ksi, axis=0)
        denominator_sum: np.ndarray = np.sum(gamma, axis=0)
        return numerator_sum, denominator_sum

    def observation_log_probability(self, z: List[Any]) -> float:
        """Run forward algorithm to compute log probability of observation."""
        self.forward_algorithm(z)
        return -np.sum(np.log(self.c))

    def reestimation(self, zs: List[List[Any]], n: int) -> np.ndarray:
        """Run Baum-Welch for a specified number of iterations.
        
        Parameters:
        ---
        zs -- List of observation sequences.
        n -- Number of iterations.
        """
        try:
            hasattr(self, "baum_welch")
        except:
            raise NotImplementedError(
                'Class must implement some form of the Baum-Welch algorithm. The method must be named "baum_welch".'
            )
        history: List[float] = []
        initial_log_probability: float = np.mean(
            list(map(self.observation_log_probability, zs))
        )
        history.append(initial_log_probability)
        for _ in range(n):
            self.baum_welch(zs)
            current_log_probability: float = np.mean(
                list(map(self.observation_log_probability, zs))
            )
            history.append(current_log_probability)

        return np.array(history)


class DiscreteEmissionProbability:
    """Class for representing probabilities for discrete observations.
    The initial argument should be a function that takes two arguments and returns the probability of
    the symbol given in the first argument when in the state given by the second argument.

    A symbol refers to a single element from the space of a finite number of possible obserservations.

    Parameters:
    ---
    emission_probability -- A function that returns the probability of symbol given state.
    states -- A list of all states in state space.
    symbols -- A list of all symbols that can be observed. 
    """

    def __init__(
        self,
        emission_probability: Callable[[Any], float],
        states: List[Any],
        symbols: List[Any],
        enable_warnings=False,
    ):
        self.enable_warnings = enable_warnings
        self.symbols = symbols
        self.states = states
        self.state_ids = np.arange(self.M)
        #This fails if symbols aren't hashable
        #Surely this can be avoided
        self.symbol_id_dictionary: Dict[Any, int] = {
            k: v for k, v in zip(self.symbols, range(self.K))
        }
        self.l_function: Callable[[Any], float] = emission_probability
        b: np.ndarray = np.array(
            list(
                map(
                    lambda x: self.l_function(x[0], x[1]),
                    product(self.symbols, self.states),
                )
            )
        ).reshape(self.K, self.M)
        self.b: np.ndarray = b

    @property
    def K(self):
        return len(self.symbols)

    @property
    def M(self):
        return len(self.states)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        unscaled_b = value
        scaled_b = value / np.sum(unscaled_b, axis=0)
        if (
            not np.all(np.isclose(unscaled_b, scaled_b, atol=1e-3))
            and self.enable_warnings
        ):
            warnings.warn("Unscaled emission matrix supplied to setter.", UserWarning)
        self._b = scaled_b

    def eval_to_array(self, z: Any, x: np.ndarray):
        """Return an array where the ith element is the probability of observing the symbol in
        the ith positon of the first argument when in state identified by the state ID in the ith
        positon of the second argument."""
        return np.array(
            [
                self.b[self.symbol_id_dictionary[a], b]
                for a, b in zip([z] * x.shape[0], x)
            ]
        )

    def l(self, z: List[Any]):
        """Creates a 2-dimensional array of emission probabilities for the observations at various times, for various states.
        
        Parameters:
        ---
        z -- A list of observations of the type expected by the InitialProbability class.
        """
        N: int = len(z)
        l_array: np.ndarray = np.zeros((N, self.M))
        for n in range(N):
            l_array[n, :] = self.eval_to_array(z[n], self.state_ids)
        # This is a hacky solution to a problem that should probably be handled in different manner.
        # The underlying is issue is that zero, or close to zero, values, are propogated throughout the algorithm and leads to division by zero at later stages.
        return np.clip(l_array, a_min=1e-9, a_max=None)


class DiscreteHiddenMarkovModel(HiddenMarkovModel):
    """A class that inherits HiddenMarkovModel and has functionality specific for Hidden Markov Models
    with a finite, discrete observation space. See documentation for HiddenMarkovModel. Notable
    changes/additions is how emission_probability should be a function as expected by DiscreteEmissionProbability,
    how the constructor needs a list of all symbols, and the inclusion of methods specific to reestimation for
    discrete Hidden Markov Models.
    """

    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        emission_probability: Callable[[Any, int], float],
        initial_probability: Callable[[int], float],
        states: List[Any],
        symbols: List[Any],
        enable_warnings=False,
        frozen_mask=None,
    ):
        self.states = states
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.symbols = symbols

        self.frozen_mask = frozen_mask
        self.enable_warnings: bool = enable_warnings

        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability,
            self.states,
            enable_warnings=self.enable_warnings,
            frozen_mask=self.frozen_mask,
        )
        self.emission_probability: EmissionProbability = DiscreteEmissionProbability(
            emission_probability,
            self.states,
            self.symbols,
            enable_warnings=self.enable_warnings,
        )
        self.initial_probability: InitialProbability = InitialProbability(
            initial_probability, self.states, enable_warnings=self.enable_warnings
        )

    @property
    def b(self):
        return self.emission_probability.b

    @b.setter
    def b(self, value):
        self.emission_probability.b = value

    def baum_welch(self, zs: List[List[Any]]):
        """Baum-Welch for discrete observations.
        
        Parameters:
        ---
        zs -- List of observation sequences.
        """
        P_numerators: List[np.ndarray] = []
        P_denominators: List[np.ndarray] = []
        b_numerators = []
        b_denominators = []
        pis = []

        E: int = len(zs)
        # Compute the log-probability for each of the observation sequences.
        ksis: List[np.ndarray] = []
        gammas: List[np.ndarray] = []
        log_probs_list: List[float] = []
        z: List[Any]
        for z in zs:
            self.forward_backward_algorithm(z)
            log_prob: float = -np.sum(np.log(self.c))
            ksis.append(self.ksi)
            gammas.append(self.gamma)
            log_probs_list.append(log_prob)
        log_probs: np.ndarray = np.array(log_probs_list)

        # Scaling for each observation when summing over results from multiple observations
        min_log_prob: float = np.min(log_probs)
        revised_scalings: np.ndarray = np.exp(min_log_prob - log_probs)

        revised_scaling: int
        ksi: np.ndarray
        gamma: np.ndarray
        for gamma, ksi, revised_scaling in zip(gammas, ksis, revised_scalings):
            P_numerator: np.ndarray
            P_denominator: np.ndarray
            P_numerator, P_denominator = self.calculate_inner_transition_probability_sums(
                ksi, gamma
            )
            P_numerators.append(P_numerator * revised_scaling)
            P_denominators.append(P_denominator * revised_scaling)
            b_numerator, b_denominator = self.calculate_inner_emission_probability_sums(
                z, gamma, self.symbols
            )
            b_numerators.append(b_numerator * revised_scaling)
            b_denominators.append(b_denominator * revised_scaling)

            pi = gamma[0, :]
            pis.append(pi)

        self.P = sum(P_numerators) / sum(P_denominators)[:, np.newaxis]
        self.b = sum(b_numerators) / sum(b_denominators)
        self.pi = sum(pis) / E

    @staticmethod
    def calculate_inner_emission_probability_sums(z, gamma, symbols):
        """Calculates sum in update equations for emission probabilties.
        
        Parameters:
        ---
        z -- A list of observations.
        gamma -- A 2-dimensional array containing all values for gamma. Time index on first axis, state index on second. 
        symbols -- A list of all symbols.
        """
        M: int = gamma.shape[1]
        K: int = len(symbols)
        numerator_sum: np.ndarray = np.ones((K, M))
        k: int
        o: Any
        for k, o in enumerate(symbols):
            # Indices where observation equals symbol k
            ts: np.ndarray = np.where(np.array(z) == o)
            # All zeros
            partial_gamma: np.ndarray = np.zeros(gamma.shape)
            # Assign non-zero values to rows (times) where observation equals symbol k
            partial_gamma[ts, :] = gamma[ts, :]
            # Sum over all time-steps
            numerator_sum[k, :] = np.sum(partial_gamma, axis=0)
        denominator_sum: np.ndarray = np.sum(gamma, axis=0)

        return numerator_sum, denominator_sum


class GaussianEmissionProbability:
    """Class for representing state-dependent Gaussian emission probabilties."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu: np.ndarray = mu
        self.sigma: np.ndarray = sigma
        self.state_ids = np.arange(self.M)

        def emission_probability(z, x) -> np.ndarray:
            return multivariate_normal.pdf(
                z, mean=self.mu[x, :], cov=self.sigma[x, :, :]
            )

        self.l_function: Callable[[Any, int], np.ndarray] = emission_probability

    def eval_to_array(self, z: np.ndarray, x: np.ndarray):
        # Can do apply along axis.
        return np.array([self.l_function(z, x) for z, x in zip(z, x)])

    def l(self, z: List[Any]):
        """Creates a 2-dimensional array of emission probabilities for the observations at various times, for various states.
        
        Parameters:
        ---
        z -- A list of observations of the type expected by the InitialProbability class.
        """
        N: int = len(z)
        l_array: np.ndarray = np.zeros((N, self.M))
        for n in range(N):
            l_array[n, :] = self.eval_to_array(z[n], self.state_ids)
        # This is a hacky solution to a problem that should probably be handled in different manner.
        # The underlying is issue is that zero, or close to zero, values, are propogated throughout the algorithm and leads to division by zero at later stages.
        return np.clip(l_array, a_min=1e-9, a_max=None)

    @property
    def M(self):
        return self.mu.shape[0]

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value


class GaussianHiddenMarkovModel(HiddenMarkovModel):
    """Class for representing Hidden Markov Models where the state dependency is expressed entirely through
    unique means and covariances for the different states.
    
    Parameters:
    transition_probability -- A function that takes two arguments, with each being argument existing in the state space,
    and returns the probability (not necessarily normalized) of transitioning from the first state into the second.
    initial_probability -- A function that, given a state, returns the probability of starting off in the given state.
    states -- A list of all the states in the state space.
    mu -- An array with shape (M, D), where M is the cardinality of the state space and D is the dimension of
    the observations. Each row is the initial mean for the M different states. 
    sigma  -- An array with shape (M, D, D), where the mth slice along the first axis is the initial covariance matrix.    
    """

    def __init__(
        self,
        transition_probability: Callable[[Any, Any], float],
        initial_probability: Callable[[int], float],
        states: List[Any],
        mu: list,
        sigma: list,
        enable_warnings: bool = False,
        frozen_mask: np.ndarray = None,
    ):
        self.states: List[Any] = states
        self.state_ids: np.ndarray = np.arange(self.M).astype(int)
        self.mu = mu
        self.sigma = sigma
        self.enable_warnings: bool = enable_warnings
        self.frozen_mask = frozen_mask
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability,
            self.states,
            enable_warnings=self.enable_warnings,
            frozen_mask=self.frozen_mask,
        )
        self.emission_probability: GaussianEmissionProbability = GaussianEmissionProbability(
            mu, sigma
        )
        self.initial_probability: InitialProbability = InitialProbability(
            initial_probability, self.states, enable_warnings=self.enable_warnings
        )

    @property
    def mu(self):
        return self.emission_probability.mu

    @mu.setter
    def mu(self, value):
        self.emission_probability.mu = value

    @property
    def sigma(self):
        return self.emission_probability.sigma

    @sigma.setter
    def sigma(self, value):
        self.emission_probability.sigma = value

    def baum_welch(self, zs: list):
        """Baum-Welch for hidden Markov model with Gaussian emissions.

        Parameters:
        ---
        zs -- List of observation sequences.
        """
        P_numerators: List[np.ndarray] = []
        P_denominators: List[np.ndarray] = []
        mu_numerators: List[np.ndarray] = []
        mu_denominators: List[np.ndarray] = []
        sigma_numerators: List[np.ndarray] = []
        sigma_denominators: List[np.ndarray] = []
        pis: List[np.ndarray] = []

        E: int = len(zs)

        ksis: List[np.ndarray] = []
        gammas: List[np.ndarray] = []
        log_probs_list: List[float] = []
        z: List[Any]
        for z in zs:
            self.forward_backward_algorithm(z)
            log_prob: float = -np.sum(np.log(self.c))
            ksis.append(self.ksi)
            gammas.append(self.gamma)
            log_probs_list.append(log_prob)
        log_probs: np.ndarray = np.array(log_probs_list)

        # Scaling for each observation when summing over results from multiple observations
        min_log_prob: float = np.min(log_probs)
        revised_scalings: np.ndarray = np.exp(min_log_prob - log_probs)

        revised_scaling: int
        ksi: np.ndarray
        gamma: np.ndarray
        for z, gamma, ksi, revised_scaling in zip(zs, gammas, ksis, revised_scalings):
            P_numerator, P_denominator = self.calculate_inner_transition_probability_sums(
                ksi, gamma
            )
            P_numerators.append(P_numerator * revised_scaling)
            P_denominators.append(P_denominator * revised_scaling)

            mu_numerator, mu_denominator = self.calculate_mu(z, gamma)
            mu_numerators.append(mu_numerator)
            mu_denominators.append(mu_denominator)

            sigma_numerator, sigma_denominator = self.calculate_sigma(
                z, self.emission_probability.mu, gamma
            )
            sigma_numerators.append(sigma_numerator)
            sigma_denominators.append(sigma_denominator)

            pi = gamma[0, :]
            pis.append(pi)

        self.P = sum(P_numerators) / sum(P_denominators)[:, np.newaxis]
        self.mu = (sum(mu_numerators) / (sum(mu_denominators)[:, np.newaxis])).reshape(
            self.M, -1
        )
        self.sigma = (
            sum(sigma_numerators) / sum(sigma_denominators)[:, np.newaxis, np.newaxis]
        )
        self.sigma = self.sigma + 1e-1 * np.eye(self.sigma.shape[1])
        self.pi = sum(pis) / E

    @staticmethod
    def calculate_mu(z: list, gamma: np.ndarray):
        """Calculate the kernel of the outer sum in the update equations for mu."""
        z_array: np.ndarray = np.array(z)
        mu_numerator: np.ndarray = np.einsum("ij,ik->kj", z_array, gamma)
        mu_denominator: np.ndarray = np.sum(gamma, axis=0)

        return mu_numerator, mu_denominator

    @staticmethod
    def calculate_sigma(z: List[Any], mu: np.ndarray, gamma: np.ndarray):
        z_array: np.ndarray = np.array(z).reshape(gamma.shape[0], -1)
        T: int = z_array.shape[0]
        D: int = z_array.shape[1]
        M: int = gamma.shape[1]
        sigma: List[float] = []
        m: int
        for m in range(M):
            comps = []
            for t in range(T):
                arr = (z_array[t, :] - mu[m, :]).reshape(D, 1)
                comps.append(gamma[t, m] * np.matmul(arr, arr.T))
            comps = np.array(comps)
            assert comps.shape == (T, D, D)
            sum_over_t = np.sum(comps, axis=0)
            assert sum_over_t.shape == (D, D)
            sigma.append(sum_over_t)

        sigma_numerator = np.array(sigma)
        sigma_denominator = np.sum(gamma, axis=0)
        return sigma_numerator, sigma_denominator
