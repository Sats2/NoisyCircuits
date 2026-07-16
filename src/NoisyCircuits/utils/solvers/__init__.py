# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

import importlib
from types import ModuleType
from typing import Literal


SolverName = Literal["pennylane", "qulacs", "qiskit", "custom"]

def load_solver(name: SolverName)->ModuleType:
    try:
        return importlib.import_module(f"NoisyCircuits.utils.{name}")
    except ImportError as e:
        raise ImportError(f"Failed to load {name} backend. Ensure dependencies are installed in the environment.") from e