from jax.tree_util import tree_flatten
import jax.numpy as jnp
import numpy as np


def tree_stack(trees: list):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    Copied from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [
        # Don't eagerly convert to JAX arrays if they're NumPy arrays
        jnp.stack(l) if isinstance(l[0], jnp.ndarray) else np.stack(l)
        for l in grouped_leaves
    ]   # type: ignore
    return treedef_list[0].unflatten(result_leaves)
