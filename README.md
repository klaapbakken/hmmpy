# General framework for working with Hidden Markov Models in Python

A generic HMM (`HiddenMarkovModel`) implements Viterbi, forward algorithm and backward algorithm for arbitrary state spaces, emission probabilities and transition probabilties.

In addition, Baum-Welch (parameter reestimation through Expectation-Maximization), is implemented for arbitrary state spaces and transition probabilities, but with either Gaussian emission probabilties (`GaussianHiddenMarkovModel`) or discrete emission probabilities (`DiscreteHiddenMarkovModel`).

Binder notebooks with usage examples will be provided at a later stage. 
