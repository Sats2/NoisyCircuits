import importlib
from types import ModuleType
from typing import Literal


SolverName = Literal["pennylane", "qulacs"]

def load_solver(name: SolverName)->ModuleType:
    try:
        return importlib.import_module(f"NoisyCircuits.utils.{name}")
    except ImportError as e:
        raise ImportError(f"Failed to load {name} backend. Ensure dependencies are installed in the environment.") from e