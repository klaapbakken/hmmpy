import numpy as np  # Arrays
from numpy import ma  # To handle logarithm if small values
from numpy.linalg import det  # For use with the multivariate normal covariance matrix

from tqdm import trange  # Keeping track of reestimation

from scipy.stats import multivariate_normal  # For use in Gaussian HMMs

from itertools import product
from functools import reduce
from functools import partial

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
    """Class for representing and evaluating initial probabilties."""

    def __init__(
        self, initial_probability, states, enable_warnings=False,
    ):
        """For initializing an object of type InitialProbability.
        
        Parameters:
        ---
        initial_probability -- A function, taking a single argument, with the argument being
        an element present in the "states" argument, that returns the probability of starting of the supplied state.
        states -- A list of all states in the state space.
        """
        self.states = states
        self.state_ids = np.arange(self.M)
        self.pi_function = initial_probability
        self.enable_warnings = enable_warnings

    def eval_to_array(self, pi_function, x):
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
        self, transition_probability, states, enable_warnings=False,
    ):
        self.states = states
        self.state_ids = np.arange(self.M).astype(int)
        self.enable_warnings = enable_warnings
        self.P_function = transition_probability

    def eval_to_array(self, P_function, x, y):
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
        states_repeated = np.repeat(self.state_ids, self.M)
        states_tiled = np.tile(self.state_ids, self.M)
        self.P = self.eval_to_array(value, states_repeated, states_tiled).reshape(
            self.M, self.M
        )

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        unscaled_P = value
        sum_unscaled_P = np.sum(unscaled_P, axis=1)
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
        self, emission_probability, states, enable_warnings=False,
    ):
        self.states = states
        self.state_ids = np.arange(self.M).astype(int)
        self.l_function = emission_probability
        self.enable_warnings = enable_warnings

    def eval_to_array(self, z, x):
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

    def l(self, z):
        """Creates a 2-dimensional array of emission probabilities for the observations at various times, for various states.
        
        Parameters:
        ---
        z -- A list of observations of the type expected by the InitialProbability class.
        """
        N = len(z)
        l_array = np.zeros((N, self.M))
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
        transition_probability,
        emission_probability,
        initial_probability,
        states,
        enable_warnings: bool = False,
        update_matrix=None,
    ):
        self.states = states
        self.state_ids = np.arange(self.M).astype(int)
        self.enable_warnings: bool = enable_warnings
        self.update_matrix = update_matrix
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability, self.states, enable_warnings=self.enable_warnings
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

    def viterbi(self, z):
        """Run Viterbi in order to obtain the state sequence that maximizes the posterior probability.
        In other words, the argmax of the probability of a state sequence conditioned the observed sequence.
        A wrapper around the internals.

        Parameters:
        ---
        z -- List of observations.
        """
        return self.log_viterbi_internals(z, self.P, self.l(z), self.pi)

    @staticmethod
    def viterbi_internals(z, P, l, pi):
        """Simple, self-contained, straight-forward implementation of Viterbi. 
        Should probably not be used, use log_viterbi_internals instead.
        
        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        pi -- The initial probabilities.
        l -- The emission probabilities.
        """
        N = len(z)
        assert pi.shape[0] == l.shape[1]
        M = pi.shape[0]

        delta = np.zeros((N, M))
        phi = np.zeros((N, M))

        delta[0, :] = pi * l[0, :]
        phi[0, :] = 0

        for n in np.arange(1, N):
            # Multiply delta by each column in P
            # In resulting matrix, for each column, find max entry
            delta[n, :] = l[n, :] * np.max((delta[n - 1, :] * P.T).T, axis=0)
            phi[n, :] = np.argmax((delta[n - 1, :] * P.T).T, axis=0)

        x_star = np.zeros((N,))
        x_star[N - 1] = np.argmax(delta[N - 1, :])

        for n in np.arange(N - 2, -1, -1):
            x_star[n] = phi[n + 1, x_star[n + 1].astype(int)]

        return x_star.astype(int)

    @staticmethod
    def log_viterbi_internals(z, P, l, pi):
        """Viterbi in log-space. More stable, i.e. resistant to almost-zero values, than the regular implementation.
        
        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        pi -- The initial probabilities.
        l -- The emission probabilities.
        """
        N = len(z)
        assert pi.shape[0] == l.shape[1]
        M = pi.shape[0]

        delta = np.zeros((N, M))
        phi = np.zeros((N, M))
        log_P = ma.log(P).filled(-np.inf)

        delta[0, :] = ma.log(pi).filled(-np.inf) + ma.log(l[0, :]).filled(-np.inf)
        phi[0, :] = 0

        for n in np.arange(1, N):
            # Multiply delta by each column in P
            # In resulting matrix, for each column, find max entry
            log_l = ma.log(l[n, :]).filled(-np.inf)
            delta[n, :] = log_l + np.max(
                (np.expand_dims(delta[n - 1, :], axis=1) + log_P), axis=0
            )
            phi[n, :] = np.argmax(
                (np.expand_dims(delta[n - 1, :], axis=1) + log_P), axis=0
            )

        q_star = np.zeros((N,))
        q_star[N - 1] = np.argmax(delta[N - 1, :])

        for n in np.arange(N - 2, -1, -1):
            q_star[n] = phi[n + 1, q_star[n + 1].astype(int)]

        return q_star.astype(int)

    def decode(self, z):
        """The Viterbi method returns an array of state ids.
        This returns the corresponding states instead.

        Parameters:
        ---
        z -- List of observations.
        """
        state_ids = self.viterbi(z)
        return list(map(lambda x: self.states[x], state_ids))

    def forward_algorithm(self, z):
        """Run the forward algorithm with object-specific arguments to the internals."""
        self.c, self.alpha = self.forward_algorithm_internals(
            z, self.P, self.l(z), self.pi
        )

    @staticmethod
    def forward_algorithm_internals(z, P, l, pi):
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
        N = len(z)
        assert pi.shape[0] == l.shape[1]
        M = pi.shape[0]

        alpha = np.zeros((N, M))
        c = np.zeros((N,))

        alpha[0, :] = l[0, :] * pi
        c[0] = np.reciprocal(np.sum(alpha[0, :]))
        alpha[0, :] = alpha[0, :] * c[0]

        for n in np.arange(N - 1):
            alpha[n + 1, :] = np.sum(alpha[n, :][:, np.newaxis] * P, axis=0) * (
                l[n + 1, :]
            )
            c[n + 1] = np.reciprocal(np.sum(alpha[n + 1, :]))
            alpha[n + 1, :] = alpha[n + 1, :] * c[n + 1]

        return c, alpha

    def backward_algorithm(self, z):
        """Wrapper around the backward algorithm, calling the internals with object-specific attributes.
        
        Parameters:
        ---
        z -- List of observations.
        """
        assert hasattr(self, "c"), "Run forward algorithm first!"
        self.beta = self.backward_algorithm_internals(
            z, self.P, self.l(z), self.pi, self.c
        )

    @staticmethod
    def backward_algorithm_internals(z, P, l, pi, c):
        """The actual implementation of the backward-algorithm. Returns beta, an array.
        See Rabiner's "Fundamentals of Speech Processing" or similar resource.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        l -- The emission probabilities.
        pi -- The initial probabilities.
        """
        N = len(z)
        assert pi.shape[0] == l.shape[1]
        M = pi.shape[0]

        beta = np.zeros((N, M))
        beta[N - 1, :] = 1 * c[N - 1]

        for n in np.arange(N - 2, -1, -1):
            b = l[n + 1, :]

            beta[n, :] = np.sum(P * b * beta[n + 1, :], axis=1) * c[n]

        return beta

    def forward_backward_algorithm(self, z):
        """Runs both forward and backward algorithm in correct order and computes ksi and gamma.
        See Rabiner for information on interpretation of these values. 
        
        Parameters:
        ---
        z -- List of observations.
        """
        self.forward_algorithm(z)
        self.backward_algorithm(z)

        self.gamma = self.calculate_gamma(self.alpha, self.beta)
        self.ksi = self.calculate_ksi(z, self.P, self.l(z), self.alpha, self.beta)

    @staticmethod
    def calculate_ksi(z, P, l, alpha, beta):
        """Compute ksi. The interpretation of ksi is the expected number of transitions from state i to j at time t.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        l -- The emission probabilities.
        pi -- The initial probabilities.
        """
        N = len(z)
        assert l.shape[0] == alpha.shape[0] == beta.shape[0] == N
        assert l.shape[1] == alpha.shape[1] == beta.shape[1]
        M = alpha.shape[1]

        ksi = np.zeros((N - 1, M, M))
        for n in range(N - 1):
            b = l[n + 1, :]
            ksi[n, :, :] = (P * b * beta[n + 1, :]) * alpha[n, :][:, np.newaxis]
            ksi[n, :, :] = ksi[n, :, :] / np.sum(ksi[n, :, :])

        return ksi

    @staticmethod
    def calculate_gamma(alpha, beta):
        """Compute gamma. The interpretation of gamma is the expected number of transitions away from state i at time t.

        Parameters:
        ---
        z -- List of observations.
        P -- The transition matrix.
        l -- The emission probabilities.
        pi -- The initial probabilities.
        """
        alpha_beta_product = alpha * beta
        sum_over_all_states = np.sum(alpha_beta_product, axis=1)
        gamma = alpha_beta_product / sum_over_all_states[:, np.newaxis]
        return gamma

    def baum_welch(self, zs):
        """Baum-Welch algorithm. Expectation-maximization. Updates the parameters to increase the expected log-probability of the observation sequence(s).

        Parameters:
        ---
        zs -- A list of observation sequences. 
        """
        P_numerators_sum = np.zeros((self.M, self.M))
        P_denominators_sum = np.zeros((self.M,))
        pis_sum = np.zeros((self.M,))

        # Compute the log-probability for each of the observation sequences.
        E = len(zs)
        for z in zs:
            self.forward_backward_algorithm(z)
            (
                P_numerator,
                P_denominator,
            ) = self.calculate_inner_transition_probability_sums(self.ksi, self.gamma)
            P_numerators_sum += P_numerator
            P_denominators_sum += P_denominator
            pi = self.gamma[0, :]
            pis_sum += pi

        if self.update_matrix is not None:
            previous_P = self.P
            new_P = P_numerators_sum / P_denominators_sum[:, np.newaxis]
            P = previous_P
            indices_to_update = self.update_matrix.nonzero()
            row_indices, column_indices = indices_to_update
            P[row_indices, column_indices] = new_P[row_indices, column_indices]
        else:
            P = P_numerators_sum / P_denominators_sum[:, np.newaxis]
        self.P = P
        self.pi = pis_sum / E

    @staticmethod
    def calculate_inner_transition_probability_sums(ksi, gamma):
        """Computing the sums required for the updated transition probabiltites.
        
        Parameters:
        ---
        ksi -- 3-dimensional array. First axis is time index, second and third are state indices.
        gamma -- 2-dimensional array. First axis is time index, second is state index. 
        """
        numerator_sum = np.sum(ksi, axis=0)
        denominator_sum = np.sum(gamma[:-1, :], axis=0)
        return numerator_sum, denominator_sum

    def observation_log_probability(self, z):
        """Run forward algorithm to compute log probability of observation."""
        self.forward_algorithm(z)
        return -np.sum(np.log(self.c))

    def reestimation(self, zs, n):
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
        history = []
        initial_log_probability = np.mean(
            list(map(self.observation_log_probability, zs))
        )
        history.append(initial_log_probability)
        print(f"Running {n} iterations of Baum-Welch")
        for _ in trange(n):
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
        self, emission_probability, states, symbols, enable_warnings=False,
    ):
        self.enable_warnings = enable_warnings
        self.symbols = symbols
        self.states = states
        self.state_ids = np.arange(self.M)
        # This fails if symbols aren't hashable
        # Surely this can be avoided
        self.symbol_id_dictionary = {k: v for k, v in zip(self.symbols, range(self.K))}
        self.l_function = emission_probability
        b = np.array(
            list(
                map(
                    lambda x: self.l_function(x[0], x[1]),
                    product(self.symbols, self.states),
                )
            )
        ).reshape(self.K, self.M)
        self.b = b

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

    def eval_to_array(self, z, x):
        """Return an array where the ith element is the probability of observing the symbol in
        the ith positon of the first argument when in state identified by the state ID in the ith
        positon of the second argument."""
        return np.array(
            [
                self.b[self.symbol_id_dictionary[a], b]
                for a, b in zip([z] * x.shape[0], x)
            ]
        )

    def l(self, z):
        """Creates a 2-dimensional array of emission probabilities for the observations at various times, for various states.
        
        Parameters:
        ---
        z -- A list of observations of the type expected by the InitialProbability class.
        """
        N = len(z)
        l_array = np.zeros((N, self.M))
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
        transition_probability,
        emission_probability,
        initial_probability,
        states,
        symbols,
        enable_warnings=False,
        update_matrix=None,
    ):
        self.states = states
        self.state_ids = np.arange(self.M).astype(int)
        self.symbols = symbols

        self.update_matrix = update_matrix
        self.enable_warnings: bool = enable_warnings

        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability, self.states, enable_warnings=self.enable_warnings,
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

    def baum_welch(self, zs):
        """Baum-Welch for discrete observations.
        
        Parameters:
        ---
        zs -- List of observation sequences.
        """
        P_numerators_sum = np.zeros((self.M, self.M))
        P_denominators_sum = np.zeros((self.M,))
        b_numerators_sum = np.zeros((len(self.symbols), self.M))
        b_denominators_sum = np.zeros((self.M,))
        pis_sum = np.zeros((self.M,))

        E = len(zs)
        for z in zs:
            self.forward_backward_algorithm(z)
            (
                P_numerator,
                P_denominator,
            ) = self.calculate_inner_transition_probability_sums(self.ksi, self.gamma)
            P_numerators_sum += P_numerator
            P_denominators_sum += P_denominator

            b_numerator, b_denominator = self.calculate_inner_emission_probability_sums(
                z, self.gamma, self.symbols
            )
            b_numerators_sum += b_numerator
            b_denominators_sum += b_denominator

            pi = self.gamma[0, :]
            pis_sum += pi

        if self.update_matrix is not None:
            previous_P = self.P
            new_P = P_numerators_sum / P_denominators_sum[:, np.newaxis]
            P = previous_P
            indices_to_update = self.update_matrix.nonzero()
            row_indices, column_indices = indices_to_update
            P[row_indices, column_indices] = new_P[row_indices, column_indices]
        else:
            P = P_numerators_sum / P_denominators_sum[:, np.newaxis]

        self.P = P
        self.b = b_numerators_sum / b_denominators_sum
        self.pi = pis_sum / E

    @staticmethod
    def calculate_inner_emission_probability_sums(z, gamma, symbols):
        """Calculates sum in update equations for emission probabilties.
        
        Parameters:
        ---
        z -- A list of observations.
        gamma -- A 2-dimensional array containing all values for gamma. Time index on first axis, state index on second. 
        symbols -- A list of all symbols.
        """
        M = gamma.shape[1]
        K = len(symbols)
        numerator_sum = np.ones((K, M))
        for k, o in enumerate(symbols):
            # Indices where observation equals symbol k
            z_is_k = np.array(list(map(lambda x: x == o, z))).astype(int)
            # All zeros
            partial_gamma = gamma * z_is_k[:, np.newaxis]
            # Assign non-zero values to rows (times) where observation equals symbol k
            # Sum over all time-steps
            numerator_sum[k, :] = np.sum(partial_gamma, axis=0)
        denominator_sum = np.sum(gamma, axis=0)

        return numerator_sum, denominator_sum


class GaussianEmissionProbability:
    """Class for representing state-dependent Gaussian emission probabilties."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.state_ids = np.arange(self.M)

        def emission_probability(z, x):
            return multivariate_normal.pdf(
                z, mean=self.mu[x, :], cov=self.sigma[x, :, :]
            )

        self.l_function = emission_probability

    def eval_to_array(self, z, x):
        # Can do apply along axis.
        return np.array([self.l_function(z, x) for z, x in zip([z] * x.shape[0], x)])

    def l(self, z):
        """Creates a 2-dimensional array of emission probabilities for the observations at various times, for various states.
        
        Parameters:
        ---
        z -- A list of observations of the type expected by the InitialProbability class.
        """
        N = len(z)
        l_array = np.zeros((N, self.M))
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
        transition_probability,
        initial_probability,
        states,
        mu: list,
        sigma: list,
        enable_warnings: bool = False,
        update_matrix=None,
    ):
        self.states = states
        self.state_ids = np.arange(self.M).astype(int)
        self.enable_warnings: bool = enable_warnings
        self.update_matrix = update_matrix
        self.transition_probability: TransitionProbability = TransitionProbability(
            transition_probability, self.states, enable_warnings=self.enable_warnings
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
        D = len(zs[0][0])
        for z in zs:
            assert all(map(lambda x: len(x) == D, z))
        P_numerators_sum = np.zeros((self.M, self.M))
        P_denominators_sum = np.zeros((self.M,))
        pis_sum = np.zeros((self.M,))
        mu_numerators_sum = np.zeros((self.M, D))
        mu_denominators_sum = np.zeros((self.M,))
        sigma_numerators_sum = np.zeros((self.M, D, D))
        sigma_denominators_sum = np.zeros((self.M,))

        E = len(zs)
        for z in zs:
            self.forward_backward_algorithm(z)
            (
                P_numerator,
                P_denominator,
            ) = self.calculate_inner_transition_probability_sums(self.ksi, self.gamma)
            P_numerators_sum += P_numerator
            P_denominators_sum += P_denominator
            pi = self.gamma[0, :]
            pis_sum += pi

            mu_numerator, mu_denominator = self.calculate_mu(z, self.gamma)
            mu_numerators_sum += mu_numerator
            mu_denominators_sum += mu_denominator

            sigma_numerator, sigma_denominator = self.calculate_sigma(
                z, self.emission_probability.mu, self.gamma
            )
            sigma_numerators_sum += sigma_numerator
            sigma_denominators_sum += sigma_denominator

        if self.update_matrix is not None:
            previous_P = self.P
            new_P = P_numerators_sum / P_denominators_sum[:, np.newaxis]
            P = previous_P
            indices_to_update = self.update_matrix.nonzero()
            row_indices, column_indices = indices_to_update
            P[row_indices, column_indices] = new_P[row_indices, column_indices]
        else:
            P = P_numerators_sum / P_denominators_sum[:, np.newaxis]
        self.P = P
        self.pi = pis_sum / E
        self.mu = (mu_numerators_sum / mu_denominators_sum[:, np.newaxis]).reshape(
            self.M, -1
        )
        self.sigma = (
            sigma_numerators_sum / sigma_denominators_sum[:, np.newaxis, np.newaxis]
        )
        self.sigma = self.sigma + 1e-1 * np.eye(self.sigma.shape[1])

    @staticmethod
    def calculate_mu(z: list, gamma):
        """Calculate the kernel of the outer sum in the update equations for mu."""
        z_array = np.array(z)
        mu_numerator = np.einsum("ij,ik->kj", z_array, gamma)
        mu_denominator = np.sum(gamma, axis=0)

        return mu_numerator, mu_denominator

    @staticmethod
    def calculate_sigma(z, mu, gamma):
        z_array = np.array(z).reshape(gamma.shape[0], -1)
        T = z_array.shape[0]
        D = z_array.shape[1]
        M = gamma.shape[1]
        sigma = []
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
