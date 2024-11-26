<img src="imgs/circuit.svg#gh-light-mode-only" width="100%">
<img src="imgs/circuit_white.svg#gh-dark-mode-only" width="100%">

# Nonlinear Quantum Computation using Amplified Encodings

*Matthias Deiml, Daniel Peterseim*

Supplementary material to the paper \
https://arxiv.org/abs/2411.16435

> This paper presents a novel framework for high-dimensional nonlinear quantum computation that exploits tensor products of amplified vector and matrix encodings to efficiently evaluate multivariate polynomials. The approach enables the solution of nonlinear equations by quantum implementations of the fixed-point iteration and Newton's method, with quantitative runtime bounds derived in terms of the error tolerance. These results show that a quantum advantage, characterized by a logarithmic scaling of complexity with the dimension of the problem, is preserved. While Newton's method achieves near-optimal theoretical complexity, the fixed-point iteration already shows practical feasibility, as demonstrated by numerical experiments solving simple nonlinear problems on existing quantum devices. By bridging theoretical advances with practical implementation, the framework of amplified encodings offers a new path to nonlinear quantum algorithms. 

Code and the corresponding output for the fixed-point iteration is contained in the jupyter notebook [`experiment.ipynb`](experiment.ipynb). Code for the Newton's method is contained in [`newton.py`](newton.py).
