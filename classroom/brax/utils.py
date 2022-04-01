"""Copied from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75"""

from brax import jumpy as jp, QP
from jax.tree_util import tree_flatten
from typing import NamedTuple
import numpy as np


class BraxClip(NamedTuple):
    states: QP
    actions: np.ndarray

    def __len__(self):
        """Returns the number of timesteps in the clip."""
        return len(self.actions)


def tree_stack(trees: list):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jp.stack(l) for l in grouped_leaves]   # type: ignore
    return treedef_list[0].unflatten(result_leaves)
