from typing import Any, Callable


def expose(func: Callable) -> Callable:
    """Decorator for exposing a function as an API callable from the client."""
    func.exposed = True
    return func
