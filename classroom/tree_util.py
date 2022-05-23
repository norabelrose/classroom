"""
Utility functions for manipulating pytrees (recursive collections of arrays or tensors)
without depending on JAX, and in a type-safe way.
"""
from typing import (
    Any, Callable, Iterable, Protocol, Sequence, TypeVar, Union,
    cast, overload, runtime_checkable
)
import numpy as np


@runtime_checkable
class TensorLike(Protocol):
    """Any object implementing the __array__ protocol."""
    def __array__(self) -> np.ndarray:
        ...


# Define pytree types recursively- this works for Pylance but unfortunately not MyPy
AnyNumber = bool | complex | float | int | np.number
AnyLeaf = Union[AnyNumber, TensorLike]
AnyTree = Union[AnyLeaf, dict[Any, 'AnyTree'], list['AnyTree'], tuple['AnyTree', ...]]

LeafInput = TypeVar('LeafInput', bool, complex, float, int, np.number, TensorLike)
LeafOutput = TypeVar('LeafOutput')


# Hacky workaround for the lack of higher-order types
@overload
def pytree_map(func: Callable[[LeafInput], LeafOutput], tree: dict[Any, LeafInput]) -> dict[Any, LeafOutput]:
    ...

@overload
def pytree_map(func: Callable, tree: dict[Any, AnyTree], strict: bool = True) -> dict[Any, AnyTree]:
    ...

@overload
def pytree_map(func: Callable[[LeafInput], LeafOutput], tree: list[LeafInput]) -> list[LeafOutput]:
    ...

@overload
def pytree_map(func: Callable, tree: list[AnyTree], strict: bool = True) -> list[AnyTree]:
    ...

@overload
def pytree_map(func: Callable[[LeafInput], LeafOutput], tree: tuple[LeafInput, ...]) -> tuple[LeafOutput, ...]:
    ...

@overload
def pytree_map(func: Callable, tree: tuple[AnyTree, ...], strict: bool = True) -> tuple[AnyTree, ...]:
    ...

@overload
def pytree_map(func: Callable[[LeafInput], LeafOutput], tree: LeafInput) -> LeafOutput:
    ...


def pytree_map(func: Callable[[LeafInput], Any], tree: AnyTree, strict: bool = True):
    """
    Recursively apply a function to all tensors in a pytree, returning the results
    in a new pytree with the same structure. Non-tensor leaves are copied.
    """
    # Stopping condition
    if isinstance(tree, (AnyNumber, TensorLike)):
        return func(cast(LeafInput, tree))
    
    # Recursive case
    if isinstance(tree, dict):
        return {k: pytree_map(func, cast(LeafInput, v)) for k, v in tree.items()}
    
    if isinstance(tree, list):
        return [pytree_map(func, cast(LeafInput, v)) for v in tree]
    
    if isinstance(tree, tuple):
        return tuple(pytree_map(func, cast(LeafInput, v)) for v in tree)
    
    if strict:
        raise TypeError(f"Found leaf '{tree}' of unsupported type '{type(tree).__name__}'- use `strict=False` to ignore")
    else:
        return tree


@overload
def pytree_flatten(tree: dict[Any, LeafInput]) -> Iterable[LeafInput]:
    ...

@overload
def pytree_flatten(tree: list[LeafInput]) -> Iterable[LeafInput]:
    ...

@overload
def pytree_flatten(tree: tuple[LeafInput, ...]) -> Iterable[LeafInput]:
    ...

@overload
def pytree_flatten(tree: AnyTree) -> Iterable[AnyLeaf]:
    ...


def pytree_flatten(tree: AnyTree) -> Iterable[AnyLeaf]:
    """Recursively iterate over all tensors in a pytree, in topological order."""
    # Stopping condition
    if isinstance(tree, AnyLeaf):
        yield tree
    
    # Recursive case
    elif isinstance(tree, dict):
        for elem in tree.values():
            yield from pytree_flatten(elem)
    
    elif isinstance(tree, Sequence):
        for elem in tree:
            yield from pytree_flatten(elem)


def pytree_stack(trees: Sequence, dim: int = 0, stack_fn: Callable[[Sequence, int], TensorLike] = np.stack) -> AnyTree:
    """
    Stack pytrees along a given dimension, returning a new pytree with the same structure.
    All pytrees are expected to have the same structure; undefined behavior will occur if
    this is not the case.
    """
    leaf_iter = (stack_fn(seq, dim) for seq in zip(*(pytree_flatten(tree) for tree in trees)))
    try:
        return pytree_map(lambda _: next(leaf_iter), trees[0])  # type: ignore
    except RuntimeError as e:
        # Calling next() on an exhausted generator raises a RuntimeError, annoyingly
        if 'StopIteration' in str(e):
            raise TypeError("All pytrees must have the same structure") from e
        else:
            raise
