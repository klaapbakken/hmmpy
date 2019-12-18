![PyPI - License](https://img.shields.io/pypi/l/hmmpy?style=for-the-badge)

![PyPI](https://img.shields.io/pypi/v/hmmpy?style=for-the-badge)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klaapbakken/hmmpy/4a6cee6b5a23dafce6e8b626879324c1a12c62aa)

The Python package `hmmpy` implements three distinct classes that are intended to be exposed to the user. The classes are:
  * `HiddenMarkovModel`
  * `DiscreteHiddenMarkovModel`
  * `GaussianHiddenMarkovModel`
  
The classes differ in what they assume about the observations. The class `HiddenMarkovModel` supports any emission probability, but does not have any procedure in place for estimating parameters related to the emission probability. The two next classes solve this by respectively assuming Gaussian distributions or discrete distributions for the observations. The *Baum-Welch algorithm* now enables estimation of either the discrete emission probabilities or the mean and covariance associated with the state. The input to the constructor of these three objects varies slightly. All three require that the state space is supplied as a list of states. They also require functions that represent the transition probabilities and initial probabilties. The supplied transition probability function should take two objects from the state space and return the probability of transitioning from the first object to the second. The function does not need to be normalized, i.e. sum to 1 over all states, since this is handled internally regardless. The initial probability function should return the initial probability of its only argument, which should be a state from the supplied states. The final argument depends on which object is being invoked. The object `HiddenMarkovModel` requires a function that returns the emission probability of a certain observation when given the observation and the state. The object `GaussianHiddenMarkovModel` requires an array of the initial values of the mean and covariance for the various states. The object `DiscreteHiddenMarkovModel` requires a list of "symbols", which is the observation space, and a function that returns the probability of a supplied "symbol" when given the state. 

After an object has been instantiated one has access to methods such as `decode`, which returns the estimated hidden state sequence and `reestimate`, which runs Baum-Welch a given number of times in an effort to learn the model parameters. During this procedure the internal representation of learnable parameters is updated after each iteration.

Interactive notebooks showing basic usage examples can be found be going into the `notebooks` folder via the Binder badge.
