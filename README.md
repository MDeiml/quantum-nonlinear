# Nonlinear Quantum Computation using Amplified Encodings

*Matthias Deiml, Daniel Peterseim*

Supplementary material to the paper \
(add link)

<img src="imgs/circuit.svg#gh-light-mode-only" width="100%">
<img src="imgs/circuit_white.svg#gh-dark-mode-only" width="100%">

> This paper introduces a novel framework for high-dimensional nonlinear quantum computation, using amplified encodings for vectors and matrices to efficiently evaluate multivariate polynomials.
We show that this framework can, for example, be used to solve nonlinear equations. We provide quantitative runtime bounds for quantum realizations of the fixed point iteration and Newton's method that are almost linear in the error tolerance and logarithmic in the problem dimension. Experimental tests demonstrate that simple problems can already be solved on present-day hardware.

Code and the corresponding output for the fixed-point iteration is contained in the jupyter notebook [`experiment.ipynb`](experiment.ipynb). Code for the Newton iteration is contained in [`newton.py`](newton.py).
