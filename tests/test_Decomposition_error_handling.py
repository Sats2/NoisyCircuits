import pytest
import os
import pickle
from pathlib import Path
from NoisyCircuits.QuantumCircuit import QuantumCircuit
from NoisyCircuits.utils.Decomposition import NonSquareMatrixError, ShapeMismatchError, NonUnitaryMatrixError
import numpy as np

qpus = QuantumCircuit.basis_gates_set.keys()
circuits_list = []
for qpu in qpus:
    file_path = os.path.join(Path(__file__).parent.parent, f"noise_models/Noise_Model_{qpu.capitalize()}_QPU.pkl")
    noise_model = pickle.load(open(file_path, "rb"))
    circuits_list.append(
        QuantumCircuit(
            num_qubits=2,
            noise_model=noise_model,
            backend_qpu_type=qpu,
            num_trajectories=1,
            num_cores=1,
            jsonize=True,
            verbose=False,
            threshold=0.1
        )
    )

@pytest.mark.parametrize("circuit", circuits_list)
def test_rz_gate(circuit):
    """
    Test RZ gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.RZ(theta="invalid_angle", qubit=0)
    with pytest.raises(TypeError):
        circuit.RZ(theta=3.14, qubit="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.RZ(theta=3.14, qubit=10)
    with pytest.raises(ValueError):
        circuit.RZ(theta=3.14, qubit=-1)
    with pytest.raises(TypeError):
        circuit.RZ(theta=None, qubit={})

@pytest.mark.parametrize("circuit", circuits_list)
def test_sx_gate(circuit):
    """
    Test SX gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.SX(qubit="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.SX(qubit=10)
    with pytest.raises(ValueError):
        circuit.SX(qubit=-1)
    with pytest.raises(TypeError):
        circuit.SX(qubit=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_x_gate(circuit):
    """
    Test X gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.X(qubit="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.X(qubit=10)
    with pytest.raises(ValueError):
        circuit.X(qubit=-1)
    with pytest.raises(TypeError):
        circuit.X(qubit=3.14)

@pytest.mark.parametrize("circuit", circuits_list)
def test_ecr_gate(circuit):
    """
    Test ECR gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.ECR(control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.ECR(control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.ECR(control=10, target=1)
    with pytest.raises(ValueError):
        circuit.ECR(control=0, target=10)
    with pytest.raises(ValueError):
        circuit.ECR(control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.ECR(control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_ry_gate(circuit):
    """
    Test RY gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.RY(theta="invalid_angle", qubit=0)
    with pytest.raises(TypeError):
        circuit.RY(theta=3.14, qubit="invlalid_qubit")
    with pytest.raises(TypeError):
        circuit.RY(theta=None, qubit={})
    with pytest.raises(ValueError):
        circuit.RY(theta=3.14, qubit=10)
    with pytest.raises(ValueError):
        circuit.RY(theta=3.14, qubit=-1)

@pytest.mark.parametrize("circuit", circuits_list)
def test_rx_gate(circuit):
    """
    Test RX gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.RX(theta="invalid_angle", qubit=0)
    with pytest.raises(TypeError):
        circuit.RX(theta=3.14, qubit="invlalid_qubit")
    with pytest.raises(TypeError):
        circuit.RX(theta=None, qubit={})
    with pytest.raises(ValueError):
        circuit.RX(theta=3.14, qubit=10)
    with pytest.raises(ValueError):
        circuit.RX(theta=3.14, qubit=-1)

@pytest.mark.parametrize("circuit", circuits_list)
def test_y_gate(circuit):
    """
    Test Y gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.Y(qubit="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.Y(qubit=10)
    with pytest.raises(ValueError):
        circuit.Y(qubit=-1)
    with pytest.raises(TypeError):
        circuit.Y(qubit=3.14)

@pytest.mark.parametrize("circuit", circuits_list)
def test_z_gate(circuit):
    """
    Test Z gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.Z(qubit="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.Z(qubit=10)
    with pytest.raises(ValueError):
        circuit.Z(qubit=-1)
    with pytest.raises(TypeError):
        circuit.Z(qubit=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_h_gate(circuit):
    """
    Test H gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.H(qubit="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.H(qubit=10)
    with pytest.raises(ValueError):
        circuit.H(qubit=-1)
    with pytest.raises(TypeError):
        circuit.H(qubit={})

@pytest.mark.parametrize("circuit", circuits_list)
def test_cx_gate(circuit):
    """
    Test CX gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.CX(control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.CX(control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.CX(control=10, target=1)
    with pytest.raises(ValueError):
        circuit.CX(control=0, target=10)
    with pytest.raises(ValueError):
        circuit.CX(control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.CX(control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_cz_gate(circuit):
    """
    Test CZ gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.CZ(control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.CZ(control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.CZ(control=10, target=1)
    with pytest.raises(ValueError):
        circuit.CZ(control=0, target=10)
    with pytest.raises(ValueError):
        circuit.CZ(control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.CZ(control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_cy_gate(circuit):
    """
    Test CY gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.CY(control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.CY(control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.CY(control=10, target=1)
    with pytest.raises(ValueError):
        circuit.CY(control=0, target=10)
    with pytest.raises(ValueError):
        circuit.CY(control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.CY(control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_swap_gate(circuit):
    """
    Test SWAP gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.SWAP(qubit1="invalid_qubit", qubit2=1)
    with pytest.raises(TypeError):
        circuit.SWAP(qubit1=0, qubit2="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.SWAP(qubit1=10, qubit2=1)
    with pytest.raises(ValueError):
        circuit.SWAP(qubit1=0, qubit2=10)
    with pytest.raises(ValueError):
        circuit.SWAP(qubit1=-1, qubit2=10)
    with pytest.raises(TypeError):
        circuit.SWAP(qubit1=None, qubit2=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_crx_gate(circuit):
    """
    Test CRX gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.CRX(theta="invalid_angle", control=0, target=1)
    with pytest.raises(TypeError):
        circuit.CRX(theta=3.14, control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.CRX(theta=3.14, control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.CRX(theta=3.14, control=10, target=1)
    with pytest.raises(ValueError):
        circuit.CRX(theta=3.14, control=0, target=10)
    with pytest.raises(ValueError):
        circuit.CRX(theta=3.14, control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.CRX(theta=None, control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_cry_gate(circuit):
    """
    Test CRY gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.CRY(theta="invalid_angle", control=0, target=1)
    with pytest.raises(TypeError):
        circuit.CRY(theta=3.14, control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.CRY(theta=3.14, control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.CRY(theta=3.14, control=10, target=1)
    with pytest.raises(ValueError):
        circuit.CRY(theta=3.14, control=0, target=10)
    with pytest.raises(ValueError):
        circuit.CRY(theta=3.14, control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.CRY(theta=None, control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_crz_gate(circuit):
    """
    Test CRZ gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.CRZ(theta="invalid_angle", control=0, target=1)
    with pytest.raises(TypeError):
        circuit.CRZ(theta=3.14, control="invalid_qubit", target=1)
    with pytest.raises(TypeError):
        circuit.CRZ(theta=3.14, control=0, target="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.CRZ(theta=3.14, control=10, target=1)
    with pytest.raises(ValueError):
        circuit.CRZ(theta=3.14, control=0, target=10)
    with pytest.raises(ValueError):
        circuit.CRZ(theta=3.14, control=-1, target=10)
    with pytest.raises(TypeError):
        circuit.CRZ(theta=None, control=None, target=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_rzz_gate(circuit):
    """
    Test RZZ gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.RZZ(theta="invalid_angle", qubit1=0, qubit2=1)
    with pytest.raises(TypeError):
        circuit.RZZ(theta=3.14, qubit1="invalid_qubit", qubit2=1)
    with pytest.raises(TypeError):
        circuit.RZZ(theta=3.14, qubit1=0, qubit2="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.RZZ(theta=3.14, qubit1=10, qubit2=1)
    with pytest.raises(ValueError):
        circuit.RZZ(theta=3.14, qubit1=0, qubit2=10)
    with pytest.raises(ValueError):
        circuit.RZZ(theta=3.14, qubit1=-1, qubit2=10)
    with pytest.raises(TypeError):
        circuit.RZZ(theta=None, qubit1=None, qubit2=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_rxx_gate(circuit):
    """
    Test RXX gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.RXX(theta="invalid_angle", qubit1=0, qubit2=1)
    with pytest.raises(TypeError):
        circuit.RXX(theta=3.14, qubit1="invalid_qubit", qubit2=1)
    with pytest.raises(TypeError):
        circuit.RXX(theta=3.14, qubit1=0, qubit2="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.RXX(theta=3.14, qubit1=10, qubit2=1)
    with pytest.raises(ValueError):
        circuit.RXX(theta=3.14, qubit1=0, qubit2=10)
    with pytest.raises(ValueError):
        circuit.RXX(theta=3.14, qubit1=-1, qubit2=10)
    with pytest.raises(TypeError):
        circuit.RXX(theta=None, qubit1=None, qubit2=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_ryy_gate(circuit):
    """
    Test RYY gate error handling for invalid parameters for different QPU types.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.RYY(theta="invalid_angle", qubit1=0, qubit2=1)
    with pytest.raises(TypeError):
        circuit.RYY(theta=3.14, qubit1="invalid_qubit", qubit2=1)
    with pytest.raises(TypeError):
        circuit.RYY(theta=3.14, qubit1=0, qubit2="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.RYY(theta=3.14, qubit1=10, qubit2=1)
    with pytest.raises(ValueError):
        circuit.RYY(theta=3.14, qubit1=0, qubit2=10)
    with pytest.raises(ValueError):
        circuit.RYY(theta=3.14, qubit1=-1, qubit2=10)
    with pytest.raises(TypeError):
        circuit.RYY(theta=None, qubit1=None, qubit2=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_apply_swap_decomposition(circuit):
    """
    Test that applying SWAP decomposition with invalid parameters raises TypeError or ValueError.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.apply_swap_decomposition(qubit1="invalid_qubit", qubit2=1)
    with pytest.raises(TypeError):
        circuit.apply_swap_decomposition(qubit1=0, qubit2="invalid_qubit")
    with pytest.raises(ValueError):
        circuit.apply_swap_decomposition(qubit1=10, qubit2=1)
    with pytest.raises(ValueError):
        circuit.apply_swap_decomposition(qubit1=0, qubit2=10)
    with pytest.raises(ValueError):
        circuit.apply_swap_decomposition(qubit1=-1, qubit2=10)
    with pytest.raises(TypeError):
        circuit.apply_swap_decomposition(qubit1=None, qubit2=[])

@pytest.mark.parametrize("circuit", circuits_list)
def test_apply_unitary(circuit):
    """
    Test that applying a unitary with invalid parameters raises TypeError or ValueError.
    """
    circuit.refresh()
    with pytest.raises(TypeError):
        circuit.apply_unitary(unitary_matrix="not_a_matrix", qubits=[0])
    with pytest.raises(TypeError):
        circuit.apply_unitary(unitary_matrix=np.eye(2), qubits="not_a_list")
    with pytest.raises(TypeError):
        circuit.apply_unitary(unitary_matrix="not_a_matrix", qubits="not_a_list")
    with pytest.raises(TypeError):
        circuit.apply_unitary(unitary_matrix=np.eye(2), qubits=[0, "1"])
    with pytest.raises(ValueError):
        circuit.apply_unitary(unitary_matrix=np.eye(2), qubits=[10, 1])
    with pytest.raises(NonSquareMatrixError):
        matrix = np.random.normal(size=(2,3))
        circuit.apply_unitary(unitary_matrix=matrix, qubits=[0, 1])
    with pytest.raises(ShapeMismatchError):
        matrix = np.eye(3)
        circuit.apply_unitary(unitary_matrix=matrix, qubits=[0, 1])
    with pytest.raises(NonUnitaryMatrixError):
        matrix = np.array([[1, 1], [0, 1]])
        circuit.apply_unitary(unitary_matrix=matrix, qubits=[0])

for circuit in circuits_list:
    circuit.shutdown()