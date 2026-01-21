import numpy as np


def compute_marginal_probs(full_system_probs:np.ndarray[np.float64],
                           trace_qubits:list[int])->np.ndarray[np.float64]:
    """
    Function to compute the marginal probabilities by tracing out specified qubits.

    Args:
        full_system_probs (np.ndarray[np.float64]): Full system probability distribution.
        trace_qubits (list[int]): List of qubits to be traced out.

    Returns:
        np.ndarray[np.float64]: Marginal probability distribution after tracing out specified qubits.
    """
    n = int(np.log2(full_system_probs.shape[0]))
    probs_tensor = full_system_probs.reshape([2] * n)
    probs_reduced = np.sum(probs_tensor, axis=tuple(trace_qubits))
    return probs_reduced.reshape(-1)