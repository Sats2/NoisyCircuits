import pickle
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
import os
from pathlib import Path
import pytest


file_path = os.path.join(Path(__file__).parent.parent, "noise_models/Noise_Model_Heron_QPU.pkl")
noise_model = pickle.load(open(file_path, "rb"))


def test_noise_model_type():
    """
    Test that the noise_model parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        BuildModel(noise_model="invalid_noise_model")
    with pytest.raises(TypeError):
        BuildModel(noise_model=12345)
    with pytest.raises(TypeError):
        BuildModel(noise_model=["list_instead_of_model"])

def test_num_qubits_type():
    """
    Test that the num_qubits parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model, num_qubits="2")
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model, num_qubits=2.5)
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model, num_qubits=[2])

def test_invalid_num_qubits_value():
    """
    Test that invalid num_qubits values raise ValueError.
    """
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model, num_qubits=0)
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model, num_qubits=-3)

def test_threshold_type():
    """
    Test that the threshold parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model, num_qubits=2, threshold="0.01")
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model, num_qubits=2, threshold=[0.01])
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model, num_qubits=2, threshold=None)

def test_invalid_threshold_value():
    """
    Test that invalid threshold values raise ValueError.
    """
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model, num_qubits=2, threshold=-0.1)
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model, num_qubits=2, threshold=1.5)

def test_threshold_default():
    """
    Test that no exception is raised when using the default threshold.
    """
    try:
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=[["u1", "u2", "u3"], ["cx"]])
    except Exception as e:
        pytest.fail(f"Unexpected exception raised with default threshold: {e}")

def test_basis_gates_type():
    """
    Test that the basis_gates parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates="u1,u2,u3,cx")
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=12345)

def test_basis_gates_contents():
    """
    Test that the basis_gates parameter raises TypeError for invalid contents.
    """
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=["u1", "invalid_gate", "cx"])
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=["u1", 123, "cx"])
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=[["u3", 1], ["u1", None, "cx"]])
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=[["sx", "rx"], ["cx", 123]])
    with pytest.raises(TypeError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2,
                            basis_gates=[["u3", 1], ["u1", "cy", "cx"]])

def test_basis_gates_empty():
    """
    Test that the basis_gates parameter raises ValueError when empty.
    """
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model,
                            num_qubits=2
                )

def test_build_model_wrong_basis_gates():
    """
    Test that invalid basis_gates combinations raise ValueError.
    """
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model,
                   num_qubits=3,
                   basis_gates=[["u1", "u2", "u3"], ["cy"]]).build_qubit_gate_model()
    with pytest.raises(UserWarning):
        BuildModel(noise_model=noise_model,
                   num_qubits=3,
                   basis_gates=[["u1", "u2", "u3"], ["cz"]]).build_qubit_gate_model()
    with pytest.raises(UserWarning):
        BuildModel(noise_model=noise_model,
                   num_qubits=3,
                   basis_gates=[["rx", "x", "sx", "rz"], ["ecr"]]).build_qubit_gate_model()

def test_num_qubits_exceeds_available():
    """
    Test that requesting more qubits than available in the noise model raises ValueError.
    """
    with pytest.raises(ValueError):
        BuildModel(noise_model=noise_model,
                   num_qubits=250,
                   basis_gates=[["rx", "x"], ["cz"]]).build_qubit_gate_model()