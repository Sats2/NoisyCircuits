import pytest
import os
import pickle
from NoisyCircuits.QuantumCircuit import QuantumCircuit
from pathlib import Path
import pennylane as qml
import numpy as np

qpus = QuantumCircuit.basis_gates_set.keys()
circuit_list_single_qubit = []
circuit_list_double_qubit = []
for qpu in qpus:
    noise_model_path = Path(__file__).parent.parent / "noise_models" / f"Noise_Model_{qpu.capitalize()}_QPU.pkl"
    noise_model = pickle.load(open(noise_model_path, "rb"))
    circuit_list_single_qubit.append(
        QuantumCircuit(
            num_qubits=1,
            noise_model=noise_model,
            backend_qpu_type=qpu,
            num_trajectories=1,
            num_cores=1,
            jsonize=True,
            verbose=False,
            threshold=1e-3
        )
    )
    circuit_list_double_qubit.append(
        QuantumCircuit(
            num_qubits=2,
            noise_model=noise_model,
            backend_qpu_type=qpu,
            num_trajectories=1,
            num_cores=1,
            jsonize=True,
            verbose=False,
            threshold=1e-3
        )
    )

instruction_map = {
            "x": lambda q: qml.X(q),
            "sx": lambda q: qml.SX(q),
            "rz": lambda t, q: qml.RZ(t, q),
            "rx": lambda t,q: qml.RX(t, q),
            "ecr": lambda q: qml.ECR(q),
            "cz": lambda q: qml.CZ(q),
            "rzz": lambda t,q: qml.IsingZZ(t, q)
        }
instruction_map_two_qubit = {
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

def get_gate_matrix_single(instructions:list)->np.ndarray:
    """
    Helper function to generate the final matrix for decomposed single qubit gates.

    Args:
        instructions (list): List containing the circuit instructions (gate decomposition)
    
    Returns:
        (np.ndarray): Matrix operator for the gate decomposition.
    """
    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit_builder(instructions):
        for entry in instructions:
            gate_instruction = entry[0]
            qubit_added = entry[1]
            params = entry[2]
            if gate_instruction in ["rz", "rx"]:
                instruction_map[gate_instruction](params, qubit_added)
            else:
                instruction_map[gate_instruction](qubit_added)
        return qml.state()
    gate_matrix = qml.matrix(circuit_builder)(instructions)
    return gate_matrix

def get_gate_matrix_double(instructions:list)->np.ndarray:
    """
    Helper function to generate the final matrix for the decomposed double qubit gates.

    Args:
        instructions (list): List containing the circuit instructions (gate decomposition)
    
    Returns:
        (np.ndarray): Matrix operator for the gate decomposition
    """
    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit_builder(instructions):
        for entry in instructions:
            gate_instruction = entry[0]
            qubit_added = entry[1]
            params = entry[2]
            if gate_instruction in ["rz", "rx", "rzz"]:
                instruction_map[gate_instruction](params, qubit_added)
            else:
                instruction_map[gate_instruction](qubit_added)
        return qml.state()
    gate_matrix = qml.matrix(circuit_builder)(instructions)
    return gate_matrix

def get_true_matrix_double(gate:str, pair:tuple[int], theta:int|float=None)->np.ndarray:
    """
    Helper function that generates the matrix for a speicied two qubit gate for a given control-target pair.

    Args:
        gate (str): Gate being applied.
        pair (tuple[int]): [control, target] qubit tuple.
        theta (int|float, optional): Angle of rotation for the Two Qubit Gate (applicable to CRX, CRY, CRZ, RXX, RYY, RZZ gates)
    
    Returns:   
        (np.ndarray): Matrix operator for the two qubit gate.
    """
    pair = list(pair)
    if theta is None:
        params = pair
    else:
        params = [theta, pair]
    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit_builder(gate, params):
        instruction_map_two_qubit[gate](params)
        return qml.state()
    gate_matrix = qml.matrix(circuit_builder)(gate, params)
    return gate_matrix

gate_matrices = {
    "x" : np.array([[0, 1], [1, 0]]),
    "sx" : 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]]),
    "rz" : lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]]),
    "ry": lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]]),
    "rx": lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]]),
    "y": np.array([[0, -1j], [1j, 0]]),
    "z": np.array([[1, 0], [0, -1]]),
    "h": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
}

qubit_pairs = [(0, 1), (1, 0)]
angles = np.linspace(-2*np.pi, 2*np.pi, 100)

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_rz_decomposition(circuit):
    """
    Test to determine the correct decomposition for the RZ gate.
    """
    for angle in angles:
        circuit.refresh()
        circuit.RZ(theta=angle, qubit=0)
        decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
        true_matrix = gate_matrices["rz"](angle)
        assert np.allclose(true_matrix, decomp_matrix), f"Failed RZ for theta = {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_sx_decomposition(circuit):
    """
    Test to determine the correct decomposition for the SX gate.
    """
    circuit.refresh()
    circuit.SX(qubit=0)
    decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
    true_matrix = gate_matrices["sx"]
    assert np.allclose(true_matrix, decomp_matrix), f"Failed SX decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_x_decomposition(circuit):
    """
    Test to determine the correct decomposition for the X gate.
    """
    circuit.refresh()
    circuit.X(qubit=0)
    decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
    true_matrix = gate_matrices["x"]
    assert np.allclose(true_matrix, decomp_matrix), f"Failed X decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_ecr_decomposition(circuit):
    """
    Test to determine the correct decomposition for the ECR gate.
    """
    circuit.refresh()
    for pair in qubit_pairs:
        circuit.refresh()
        circuit.ECR(pair[0], pair[1])
        decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
        true_matrix = get_true_matrix_double("ecr", pair, None)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed ECR decomposition for pair {pair}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_ry_decomposition(circuit):
    """
    Test to determine the correct decomposition for the RY gate.
    """
    for angle in angles:
        circuit.refresh()
        circuit.RY(angle, qubit=0)
        decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
        true_matrix = gate_matrices["ry"](angle)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed RY decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_rx_decomposition(circuit):
    """
    Test to determine the correct decomposition for the RX gate.
    """
    for angle in angles:
        circuit.refresh()
        circuit.RX(angle, qubit=0)
        decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
        true_matrix = gate_matrices["rx"](angle)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed RX decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_y_decomposition(circuit):
    """
    Test to determine the correct decomposition for the Y gate.
    """
    circuit.refresh()
    circuit.Y(qubit=0)
    decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
    true_matrix = gate_matrices["y"]
    assert np.allclose(decomp_matrix, true_matrix), f"Failed Y decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_z_decomposition(circuit):
    """
    Test to determine the correct decomposition for the Z gate.
    """
    circuit.refresh()
    circuit.Z(qubit=0)
    decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
    true_matrix = gate_matrices["z"]
    assert np.allclose(decomp_matrix, true_matrix), f"Failed Z decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_single_qubit)
def test_h_decomposition(circuit):
    """
    Test to determine the correct decomposition for the H gate.
    """
    circuit.refresh()
    circuit.H(qubit=0)
    decomp_matrix = get_gate_matrix_single(circuit.instruction_list)
    true_matrix = gate_matrices["h"]
    assert np.allclose(decomp_matrix, true_matrix), f"Failed H decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_cx_decomposition(circuit):
    """
    Test to determine the correct decomposition for the CX gate.
    """
    for pair in qubit_pairs:
        circuit.refresh()
        circuit.CX(control=pair[0], target=pair[1])
        decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
        true_matrix = get_true_matrix_double("cx", pair, None)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed CX decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_cy_decomposition(circuit):
    """
    Test to determine the correct decomposition for the CY gate.
    """
    for pair in qubit_pairs:
        circuit.refresh()
        circuit.CY(control=pair[0], target=pair[1])
        decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
        true_matrix = get_true_matrix_double("cy", pair, None)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed CY decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_cz_decomposition(circuit):
    """
    Test to determine the correct decomposition for the CZ gate.
    """
    for pair in qubit_pairs:
        circuit.refresh()
        circuit.CZ(control=pair[0], target=pair[1])
        decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
        true_matrix = get_true_matrix_double("cz", pair, None)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed CZ decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_swap_decomposition(circuit):
    """
    Test to determine the correct decomposition for the SWAP gate.
    """
    for pair in qubit_pairs:
        circuit.refresh()
        circuit.SWAP(qubit1=pair[0], qubit2=pair[1])
        decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
        true_matrix = get_true_matrix_double("swap", pair, None)
        assert np.allclose(decomp_matrix, true_matrix), f"Failed SWAP decomposition, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_crx_decomposition(circuit):
    """
    Test to determine the correct decomposition for the CRX gate.
    """
    for pair in qubit_pairs:
        for angle in angles:
            circuit.refresh()
            circuit.CRX(theta=angle, control=pair[0], target=pair[1])
            decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
            true_matrix = get_true_matrix_double("crx", pair, angle)
            assert np.allclose(decomp_matrix, true_matrix), f"Failed CRX decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_cry_decomposition(circuit):
    """
    Test to determine the correct decomposition for the CRY gate.
    """
    for pair in qubit_pairs:
        for angle in angles:
            circuit.refresh()
            circuit.CRY(theta=angle, control=pair[0], target=pair[1])
            decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
            true_matrix = get_true_matrix_double("cry", pair, angle)
            assert np.allclose(decomp_matrix, true_matrix), f"Failed CRY decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_crz_decomposition(circuit):
    """
    Test to determine the correct decomposition for the CRZ gate.
    """
    for pair in qubit_pairs:
        for angle in angles:
            circuit.refresh()
            circuit.CRZ(theta=angle, control=pair[0], target=pair[1])
            decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
            true_matrix = get_true_matrix_double("crz", pair, angle)
            assert np.allclose(decomp_matrix, true_matrix), f"Failed CRZ decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_rxx_decomposition(circuit):
    """
    Test to determine the correct decomposition for the RXX gate.
    """
    for pair in qubit_pairs:
        for angle in angles:
            circuit.refresh()
            circuit.RXX(theta=angle, qubit1=pair[0], qubit2=pair[1])
            decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
            true_matrix = get_true_matrix_double("rxx", pair, angle)
            assert np.allclose(decomp_matrix, true_matrix), f"Failed RXX decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_ryy_decomposition(circuit):
    """
    Test to determine the correct decomposition for the RYY gate.
    """
    for pair in qubit_pairs:
        for angle in angles:
            circuit.refresh()
            circuit.RYY(theta=angle, qubit1=pair[0], qubit2=pair[1])
            decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
            true_matrix = get_true_matrix_double("ryy", pair, angle)
            assert np.allclose(decomp_matrix, true_matrix), f"Failed RYY decomposition for angle {angle}, QPU - {circuit.qpu}"

@pytest.mark.parametrize("circuit", circuit_list_double_qubit)
def test_rzz_decomposition(circuit):
    """
    Test to determine the correct decomposition for the RZZ gate.
    """
    for pair in qubit_pairs:
        for angle in angles:
            circuit.refresh()
            circuit.RZZ(theta=angle, qubit1=pair[0], qubit2=pair[1])
            decomp_matrix = get_gate_matrix_double(circuit.instruction_list)
            true_matrix = get_true_matrix_double("rzz", pair, angle)
            assert np.allclose(decomp_matrix, true_matrix), f"Failed RZZ decomposition for angle {angle}, QPU - {circuit.qpu}"

for circuit in circuit_list_single_qubit:
    circuit.shutdown()
for circuit in circuit_list_double_qubit:
    circuit.shutdown()