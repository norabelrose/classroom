from .pref_graph import PrefGraph
from scipy.optimize import Bounds, minimize
# from scipy.sparse.linalg import eigsh, inv
from scipy.special import log1p
from scipy.stats import logistic, norm, rv_continuous
from typing import Literal
import networkx as nx
import numpy as np
import warnings


def estimate_rewards(
        G: PrefGraph,
        family: Literal['bradley-terry', 'thurstone'] = 'bradley-terry',
        *,
        eps: float = 0.125,
        prior: rv_continuous | None = None,
        tol: float = 1e-5
    ):
    """
    Compute maximum a posteriori (MAP) estimates of the utilities associated with the
    nodes of a preference graph.

    Parameters
    ----------
    graph: PrefGraph
        The preference graph to estimate utilities for.
    family: Literal['bradley-terry', 'thurstone']
        The family of paired compairson model to use for estimation. Bradley-Terry
        models assume differences in utilities have a logistic distribution, whereas
        Thurstone models assume a Gaussian distribution.
    eps: float
        Laplace smoothing parameter to ensure that each node has a nonzero probability
        of being preferred to every other node.
    prior: rv_continuous | None
        A SciPy continuous probability distribution representing the prior over
        latent rewards. If None, an improper uniform prior is used.
    
    Raises
    -------
    RuntimeError
        If the inner L-BFGS-B solver fails to converge.
    """
    assert eps > 0, "Laplace smoothing parameter must be positive"
    assert G, "Graph must be non-empty"

    # The estimates for the latent utilities are coefficients of a generalized linear
    # model whose design matrix is the negative transpose of the graph incidence matrix.
    # The matrix is constructed so that multiplying it by the vector of latent
    # utilities yields a vector of length k where each entry is the difference
    # in utilities f(i) - f(j) for the corresponding edge.
    X = -nx.incidence_matrix(G, oriented=True).T
    y = np.array([G.pref_prob(a, b, eps=eps) for a, b in G.edges])

    # Start from previously computed estimates of the latent rewards if available.
    b0 = G.graph.get('rewards')
    if b0 is None or len(b0) != len(G):
        b0 = np.zeros(G.number_of_nodes())

    match family:
        case 'bradley-terry':   link = logistic     # Logistic regression
        case 'thurstone':       link = norm         # Probit model
        case other:
            raise ValueError(f"Unknown family: {other}")
    
    def loss_and_grad(b: np.ndarray):
        z = X @ b       # f(i) - f(j)
        p = link.cdf(z)

        loss = -y @ link.logcdf(z) - (1 - y) @ log1p(-p)
        grad = X.T @ (p - y)    # Reduced form of the gradient

        if prior:
            # Use a finite difference approximation to the gradient of the log
            # prior for simplicity and generality
            log_density = prior.logpdf(b)
            loss -= log_density.sum()
            grad -= (prior.logpdf(b + 1.5e-8) - log_density) / 1.5e-8
        
        return loss, grad

    result = minimize(
        loss_and_grad,
        b0,
        bounds=Bounds(*prior.support()) if prior else None,
        jac=True,
        method='L-BFGS-B',
        tol=tol
    )
    if not result.success:
        raise RuntimeError(f"Reward estimation failed to converge: {result.message}")
    
    # Cache the results for future use
    rewards = G.graph['rewards'] = result.x
    return rewards

warnings.filterwarnings('ignore', category=FutureWarning, message='incidence_matrix')