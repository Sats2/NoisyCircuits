import numpy as np


def compute_marginal_probs(full_system_probs:np.ndarray[np.float64],
                           trace_qubits:list[int])->np.ndarray[np.float64]:
    """
    Function to compute the marginal probabilities by tracing out specified qubits.

    Parameters:
    -----------
    full_system_probs : np.ndarray[np.float64]
        Full system probability distribution.
    trace_qubits : list[int]
        List of qubits to be traced out.

    Returns:
    --------
    np.ndarray[np.float64]
        Marginal probability distribution after tracing out specified qubits.
    """
    n = int(np.log2(full_system_probs.shape[0]))
    probs_tensor = full_system_probs.reshape([2] * n)
    probs_reduced = np.sum(probs_tensor, axis=tuple(trace_qubits))
    return probs_reduced.reshape(-1)


def convert_matrix_to_little_endian(matrix_list:list[np.ndarray[np.complex128]])->list[np.ndarray[np.complex128]]:
    """
    Converts the two-qubit matrices from big endian to little endian format.

    Parameters:
    -----------
    matrix_list :  list[np.ndarray[np.complex128]]
         The list of input matrices in big endian format.

    Returns:
    -------
    list[np.ndarray[np.complex128]]
        The list of output matrices in little endian format.
    """
    perm = np.array([0, 2, 1, 3])
    n = matrix_list[0].shape[0]
    result_list = []
    for matrix in matrix_list:
        result = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                result[i, j] = matrix[perm[i], perm[j]]
        result_list.append(result)
    return result_list