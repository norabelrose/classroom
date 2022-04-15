from .pref_graph import PrefGraph
from scipy.sparse.linalg import lsqr
from scipy.stats import norm
import networkx as nx
import numpy as np
import warnings


def thurstone_mle(graph: PrefGraph, eps: float = 0.1, normalize: bool = True):
    """
    Compute the maximum likelihood estimates of the utilities/rewards associated
    with the nodes of a preference graph using Case V of Thurstone (1927)'s paired
    comparisons model. This is a simplified version of the model which assumes that
    the covariance matrix is diagonal and isotropic. By default, the resulting rewards
    are normalized to unit variance.
    """
    assert 0 < eps < 0.5, "Comparison uncertainty must be in (0, 0.5)"
    assert graph, "Graph must be non-empty"

    # Sparse matrix of shape (n, k) where N is the number of nodes and k is the
    # number of edges. An entry (i, j) is 1 if node i is the source of edge j,
    # -1 if node i is the target of edge j, and 0 otherwise.
    A = -nx.incidence_matrix(graph, oriented=True).T
    y = norm.ppf(
        # The latent rewards are only identifiable if no comparisons are made with
        # certainty (the inverse Gaussian CDF isn't defined at p = 1), so we "smooth"
        # the comparisons by assuming an error rate of eps.
        np.clip(
            [graph.edges[u, v].get('confidence', 1.0) for u, v in graph.edges],
            eps, 1 - eps
        )
    )

    # Start from previously computed estimates of the latent rewards if available.
    has_rewards = 'reward' in next(iter(graph.nodes))
    b0 = [graph.nodes[node]['reward'] for node in graph.nodes] if has_rewards else None

    # The Thurstone model gives us the following system of equations:
    #   P(i preferred to j | f(i), f(j)) = norm.cdf([f(i) - f(j)] / sqrt(2))
    #       for all preferences i, j
    # where f(i) denotes the latent reward assigned to node i. Re-arranging, we get:
    #   y_ij = f(i) - f(j)
    #       where y_ij = norm.ppf(P(i preferred to j | f(i), f(j))) * sqrt(2)
    # We can now pack all the y_ij values into a vector 'y' and represent the graph
    # connectivity as an oriented incidence matrix 'A', yielding the matrix equation
    #   Ab = y
    # where 'b' is the vector of latent rewards. We can't solve this directly because
    # 'A' is rectangular, so we compute the least-squares solution.
    rewards, *_, lsq_var = lsqr(A, y, calc_var=True, x0=b0)

    # When there are long directed paths between nodes, the least-squares rewards tend
    # to have large magnitudes (e.g. 1e2, 1e4). For stable neural network training and
    # interpretability it's useful to normalize the rewards to unit variance.
    reward_var = np.var(rewards)
    if normalize:
        rewards /= (reward_var + 1e-7) ** 0.5
        lsq_var /= (reward_var + 1e-7)
        reward_var = 1
    
    # Update the node attributes with the computed rewards.
    for node, reward, stderr in zip(graph.nodes, rewards, lsq_var ** 0.5):
        graph.nodes[node].update(reward=reward, reward_stderr=stderr)
    
    return graph

# Filter annoying warnings about nx.incidence_matrix
warnings.filterwarnings('ignore', category=FutureWarning)
