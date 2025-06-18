import pennylane as qml
from pennylane import numpy as np
import json

class BuildModel:
    def __init__(self,
                 noise_model:dict):
        self.noise_model = noise_model

    def _ensure_ctpt(kraus_ops:list)->bool:
        mat = np.zeros(kraus_ops[0].shape, dtype=complex, requires_grad=False)
        for op in kraus_ops:
            mat += np.dot(op.conj().T, op)
        if not np.allclose(mat, np.eye(mat.shape[0], dtype=complex), atol=1e-8):
            raise Warning("Kraus operators do not form a complete positivity trace-preserving (CPTP) map.")
        return True
    
    def build_qubit_gate_model(self)->dict:
        gate_model = {}
        return gate_model
    
    