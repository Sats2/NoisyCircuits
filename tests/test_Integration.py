import pennylane as qml
import numpy as np
from NoisyCircuits.QuantumCircuit import QuantumCircuit
from NoisyCircuits.utils import CreateNoiseModel
import pytest
import itertools
from pathlib import Path
import pickle

qpus = QuantumCircuit.basis_gates_set.keys()
circuit_list = []
for qpu in qpus:
    if qpu == "eagle":
        file_path = Path(__file__).parent.parent / "noise_models" / f"Noise_Model_{qpu.capitalize()}_QPU.pkl"
        noise_model = pickle.load(open(file_path, "rb"))
    else:
        file_path = Path(__file__).parent.parent / "noise_models" / f"Sample_Noise_Model_{qpu.capitalize()}_QPU.csv"
        noise_model = CreateNoiseModel(calibration_data_file=str(file_path), 
                                        basis_gates=[["x", "sx", "rz", "rx"], ["cz", "rzz"]]).create_noise_model()
    for sim_backend in QuantumCircuit.available_sim_backends:
        circuit_list.append(
            (QuantumCircuit, (4, noise_model, qpu, sim_backend, 1e-6, False))
        )
num_cores = 40
np.random.seed(42)

@pytest.fixture
def circuit(request):
    cls, args = request.param
    return cls(*args)

def fidelity(p:np.ndarray, 
             q:np.ndarray)->np.float64:
    """"
    Helper function that computes the fidelity between two discrete probability distribution by means of the Hellinger Distance.

    Parameters
    ----------
    p : np.ndarray
        Probability distribution of size (n,)
    q : np.ndarray
        Probabiltiy distribution of size (n,)
    
    Returns
    -------
    np.float64
        The Hellinger Distance between the two distributions.

    Raises
    ------
    ValueError
        When the size of p and q don't match
    """
    if p.shape[0] != q.shape[0]:
        raise ValueError("Shape Mismatch between the distributions.")
    fid = 0.0
    for i in range(p.shape[0]):
        fid += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    fid = np.sqrt(fid) / np.sqrt(2)
    return fid

instruction_map = {
    "x" : lambda p, q: qml.X(q[0]),
    "sx" : lambda p, q: qml.SX(q[0]),
    "rz" : lambda p, q: qml.RZ(p, q[0]),
    "rx" : lambda p, q: qml.RX(p, q[0]),
    "ecr" : lambda p, q: qml.ECR(q),
    "cz" : lambda p, q: qml.CZ(q),
    "rzz" : lambda p, q: qml.IsingZZ(p, q)
}

def get_reference_output(
                        instructions:list,
                        measure_qubits:list,
                        return_state:bool
                        )->np.ndarray:
    """
    Function that runs a quantum circuit using a well established quantum simulator.

    Parameters
    ----------
    instructions : list
        A list of instructions, where each instruction is a tuple of the form (gate_name, qubits, parameters).
    measure_qubits : list
        A list of qubits to be measured at the end of the circuit.
    return_state : bool
        A boolean flag that indicates whether to return the final state vector or the probability distribution over the measured qubits.

    Returns
    -------
    np.ndarray
        The reference output of the circuit, either as a state vector or as a probability distribution over the measured qubits.
    """
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def run_circuit():
        """
        Executes the quantum circuit.
        """
        for entry in instructions:
            gate = entry[0]
            qubits = entry[1]
            params = entry[2]
            instruction_map[gate](params, qubits)
        if return_state:
            return qml.state()
        else:
            return qml.probs(wires=measure_qubits)
    return run_circuit()

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_ghz_probabilities(circuit):
    """
    Test that the probability output over the computational basis for the GHZ state matches the output from a well established quantum simulator.
    """
    circuit.refresh()
    circuit.H(0)
    for i in range(3):
        circuit.CX(i, i+1)
    instructions = circuit.instruction_list
    p_software = circuit.run_pure_state([0, 1, 2, 3], 1, False)
    p_reference = get_reference_output(instructions, [0, 1, 2, 3], False)
    assert np.allclose(p_software, p_reference, atol=1e-12), f"Probability distributions do not match for the GHZ state for {circuit.qpu} QPU with {circuit.sim_backend} simulator."

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_ghz_statevector(circuit):
    """
    Test that the statevector output over the computational basis for the GHZ state matches the output from a well established quantum simulator.
    """
    circuit.refresh()
    circuit.H(0)
    for i in range(3):
        circuit.CX(i, i+1)
    instructions = circuit.instruction_list
    state_software = circuit.run_pure_state([0, 1, 2, 3], 1, True)
    state_reference = get_reference_output(instructions, [0, 1, 2, 3], True)
    assert np.allclose(state_software, state_reference, atol=1e-12), f"State vectors do not match for the GHZ state for {circuit.qpu} QPU with {circuit.sim_backend} simulator."

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_ghz_parallel(circuit):
    """
    Test that the probability output over the computational basis for the GHZ state matches the output from a well established quantum simulator under parallel execution.
    """
    circuit.refresh()
    circuit.H(0)
    for i in range(3):
        circuit.CX(i, i+1)
    instructions = circuit.instruction_list
    p_software = circuit.run_pure_state([0, 1, 2, 3], num_cores, False)
    p_reference = get_reference_output(instructions, [0, 1, 2, 3], False)
    assert np.allclose(p_software, p_reference, atol=1e-12), f"Probability distributions do not match for the GHZ state for {circuit.qpu} QPU with {circuit.sim_backend} simulator when run in parallel."

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_ghz_noisy_simulation(circuit):
    """
    Test that the noise-aware simulation closely matches the density matrix output for the GHZ state.
    """
    circuit.refresh()
    circuit.H(0)
    for i in range(3):
        circuit.CX(i, i+1)
    p_software = circuit.execute([0, 1, 2, 3], 1000, num_cores)
    p_reference = circuit.run_with_density_matrix([0, 1, 2, 3], 2)
    fid = fidelity(p_software, p_reference)
    assert fid < 0.05, f"Fidelity too low for the GHZ state for {circuit.qpu} QPU with {circuit.sim_backend} simulator.\nHellinger Distance: {fid}"

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_parametrized_circuit_probabilities(circuit):
    """
    Test that the probability output over the computational basis for a parametrized quantum circuit matches the output from a well established quantum simulator.
    """
    circuit.refresh()
    for q in range(4):
        circuit.H(q)
    for _ in range(4):
        for q in range(4):
            circuit.RY(np.random.uniform(-2*np.pi, 2*np.pi), q)
        for i in range(3):
            circuit.CY(i, i+1)
    instructions = circuit.instruction_list
    p_software = circuit.run_pure_state([0, 1, 2, 3], 1, False)
    p_reference = get_reference_output(instructions, [0, 1, 2, 3], False)
    assert np.allclose(p_software, p_reference, atol=1e-12), f"Probability distributions do not match for the parametrized circuit for {circuit.qpu} QPU with {circuit.sim_backend} simulator."

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_parametrized_circuit_statevector(circuit):
    """
    Test that the statevector output over the computational basis for a parametrized quantum circuit matches the output from a well established quantum simulator.
    """
    circuit.refresh()
    for q in range(4):
        circuit.H(q)
    for _ in range(4):
        for q in range(4):
            circuit.RY(np.random.uniform(-2*np.pi, 2*np.pi), q)
        for i in range(3):
            circuit.CY(i, i+1)
    instructions = circuit.instruction_list
    p_software = circuit.run_pure_state([0, 1, 2, 3], 1, True)
    p_reference = get_reference_output(instructions, [0, 1, 2, 3], True)
    assert np.allclose(p_software, p_reference, atol=1e-12), f"Probability distributions do not match for the parametrized circuit for {circuit.qpu} QPU with {circuit.sim_backend} simulator."

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_parametrized_circuit_parallel(circuit):
    """
    Test that the probability output over the computational basis for a parametrized quantum circuit matches the output from a well established quantum simulator under parallel execution.
    """
    circuit.refresh()
    for q in range(4):
        circuit.H(q)
    for _ in range(4):
        for q in range(4):
            circuit.RY(np.random.uniform(-2*np.pi, 2*np.pi), q)
        for i in range(3):
            circuit.CY(i, i+1)
    instructions = circuit.instruction_list
    p_software = circuit.run_pure_state([0, 1, 2, 3], num_cores, False)
    p_reference = get_reference_output(instructions, [0, 1, 2, 3], False)
    assert np.allclose(p_software, p_reference, atol=1e-12), f"Probability distributions do not match for the parametrized circuit for {circuit.qpu} QPU with {circuit.sim_backend} simulator."

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_parametrized_circuit_noisy_simulation(circuit):
    """
    Test that the noise-aware simulation closely matches the density matrix output for a parametrized quantum circuit.
    """
    circuit.refresh()
    for q in range(4):
        circuit.H(q)
    for _ in range(4):
        for q in range(4):
            circuit.RY(np.random.uniform(-2*np.pi, 2*np.pi), q)
        for i in range(3):
            circuit.CY(i, i+1)
    p_software = circuit.execute([0, 1, 2, 3], 1000, num_cores)
    p_reference = circuit.run_with_density_matrix([0, 1, 2, 3], 2)
    fid = fidelity(p_software, p_reference)
    assert fid < 0.05, f"Fidelity too low for the parametrized circuit for {circuit.qpu} QPU with {circuit.sim_backend} simulator.\nHellinger Distance: {fid}"

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_shutdown(circuit):
    """
    Shutdown parallel processes and free up resources after the tests.
    """
    circuit.shutdown()