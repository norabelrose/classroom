from importlib.util import find_spec, LazyLoader, module_from_spec
import sys


def lazy_import(module_name: str, *, fail_early: bool = False):
    """
    Returns a module object which lazily loads its contents. If `fail_early` is
    `False`, then no exception is raised if the module does not exist, and a
    placeholder module is returned instead.
    """
    spec = find_spec(module_name)
    if spec is None:
        # Only raise an error if `fail_early` is True
        if fail_early:
            raise ImportError(f"Module `{module_name}` not found.")
        else:
            return FakeModule(module_name)
    
    # This just shouldn't happen
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Loader for module `{module_name}` not found.")
    
    loader = LazyLoader(loader)
    spec.loader = loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)

    return module


class FakeModule:
    """
    Placeholder for non-existent modules which raises an error as soon as it's used.
    This allows us to import classes that rely on optional dependencies in `__init__.py`
    files without worrying about whether those dependencies are installed.
    """
    def __init__(self, name: str):
        self.__name__ = name
    
    def __getattr__(self, _: str):
        raise ImportError(f"Module `{self.__name__}` not found.")
