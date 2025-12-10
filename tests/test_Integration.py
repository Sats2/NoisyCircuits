import pickle
import pennylane as qml
import numpy as np
from NoisyCircuits.QuantumCircuit import QuantumCircuit
import os
import pytest
from pathlib import Path

qpus = QuantumCircuit.basis_gates_set.keys()
circuit_list = []
for qpu in qpus:
    file_path = Path(__file__).parent.parent / "noise_models" / f"Noise_Model_{qpu.capitalize()}_QPU.pkl"
    noise_model = pickle.load(open(file_path, "rb"))
    circuit_list.append((QuantumCircuit, (4, noise_model, qpu, 200, 4, True, False, 1e-6)))

@pytest.fixture
def circuit(request):
    cls, args = request.param
    return cls(*args)

def fidelity(p:np.ndarray, 
             q:np.ndarray)->np.float64:
    """"
    Helper function that computes the fidelity between two discrete probability distribution by means of the Hellinger Distance.

    Args:
        p (np.ndarray): Probability distribution of size (n,)
        q (np.ndarray): Probabiltiy distribution of size (n,)
    
    Raises:
        ValueError: When the size of p and q don't match
    
    Returns:
        (np.float64): The Hellinger Distance between the two distributions.
    """
    if p.shape[0] != q.shape[0]:
        raise ValueError("Shape Mismatch between the distributions.")
    fid = 0.0
    for i in range(p.shape[0]):
        fid += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    fid = np.sqrt(fid) / np.sqrt(2)
    return fid

instruction_map = {
        "x": lambda q: qml.X(q),
        "sx": lambda q: qml.SX(q),
        "rz": lambda q: qml.RZ(q[0], q[1]),
        "rx": lambda q: qml.RX(q[0], q[1]),
        "h": lambda q: qml.H(q),
        "ry": lambda q: qml.RY(q[0], q[1]),
        "y": lambda q: qml.Y(q),
        "z": lambda q: qml.Z(q),
        "ecr": lambda q: qml.ECR(q),
        "cz": lambda q: qml.CZ(q),
        "cx": lambda q: qml.CNOT(q),
        "cy": lambda q: qml.CY(q),
        "swap": lambda q: qml.SWAP(q),
        "crx": lambda q: qml.CRX(q[0], q[1]),
        "cry": lambda q: qml.CRY(q[0], q[1]),
        "crz": lambda q: qml.CRZ(q[0], q[1]),
        "rzz": lambda q: qml.IsingZZ(q[0], q[1]),
        "rxx": lambda q: qml.IsingXX(q[0], q[1]),
        "ryy": lambda q: qml.IsingYY(q[0], q[1])
    }

def run_true(instructions:list)->np.ndarray:
    """
    Execute the quantum circuit without any decompositions or qubit couplings.

    Args:
        instructions (list): Instruction List to construct the circuit

    Returns:
        (np.ndarray): Probability of the quantum circuit
    """
    dev = qml.device("lightning.qubit", wires=4)
    @qml.qnode(dev)
    def _run_circuit(instructions:list):
        for entry in instructions:
            gate = entry[0]
            qubits = entry[1]
            params = entry[2]
            if params is None:
                instruction_map[gate](qubits)
            else:
                instruction_map[gate]([params, qubits])
        return qml.probs([0, 1, 2, 3])
    probs = _run_circuit(instructions)
    return probs

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_ghz_output(circuit):
    """
    Test the decomposition, gate directionality and swap sequencing with the GHZ State test.
    """
    instructions = []
    circuit.refresh()
    circuit.H(qubit=0)
    instructions.append(["h", [0], None])
    for i in range(3):
        circuit.CX(control=i, target=i+1)
        instructions.append(["cx", [i, i+1], None])
    probs_decomp = circuit.run_pure_state([0, 1, 2, 3])
    probs_true = run_true(instructions)
    circuit.shutdown()
    assert np.allclose(probs_decomp, probs_true), f"Failed GHZ Output Test for QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_mcwf_implementation_ghz(circuit):
    """
    Test fidelity closeness for the GHZ state
    """
    circuit.refresh()
    circuit.H(qubit=0)
    for i in range(3):
        circuit.CX(i, i+1)
    probs_mcwf = circuit.execute([0, 1, 2, 3])
    probs_density_matrix = circuit.run_with_density_matrix([0, 1, 2, 3])
    hellinger_distance = fidelity(probs_density_matrix, probs_mcwf)
    circuit.shutdown()
    assert (hellinger_distance < 0.05), f"Failed fidelity test for GHZ State with a fidelity of {hellinger_distance} for QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_parameterized_circuit(circuit):
    """
    Test the decomposition, gate directionality and swap sequencing with a randomized parameterized test.
    """
    angles = np.random.uniform(-2*np.pi, 2*np.pi, 16)
    instructions = []
    circuit.refresh()
    for i in range(4):
        circuit.H(qubit=i)
        instructions.append(["h", [i], None])
    for layer in range(4):
        for i in range(4):
            circuit.RY(angles[4*layer + i], i)
            instructions.append(["ry", [i], angles[4*layer+i]])
        for j in range(3):
            circuit.CY(0, j+1)
            instructions.append(["cy", [0, j+1], None])
    probs_decomp = circuit.run_pure_state([0, 1, 2, 3])
    probs_true = run_true(instructions)
    circuit.shutdown()
    assert np.allclose(probs_decomp, probs_true), f"Failed Randomized Parameterized Circuit Test for QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list, indirect=True)
def test_parameterized_circuit_fidelity(circuit):
    """
    Test the fidelity closeness a randomized parameterized test.
    """
    angles = np.random.uniform(-2*np.pi, 2*np.pi, 16)
    circuit.refresh()
    for i in range(4):
        circuit.H(qubit=i)
    for layer in range(4):
        for i in range(4):
            circuit.RY(angles[4*layer + i], i)
        for j in range(3):
            circuit.CY(0, j+1)
    probs_mcwf = circuit.execute([0, 1, 2, 3])
    probs_density_matrix = circuit.run_with_density_matrix([0, 1, 2, 3])
    circuit.shutdown()
    hellinger_distance = fidelity(probs_density_matrix, probs_mcwf)
    assert (hellinger_distance < 0.05), f"Failed Randomized Parameterized Circuit Test with Fidelity {hellinger_distance} for QPU - {circuit.qpu}"
