import os
from pathlib import Path
import pytest
import pickle
from NoisyCircuits.QuantumCircuitMPI import QuantumCircuitMPI
import subprocess
import sys


file_path = os.path.join(Path(__file__).parent.parent, "noise_models/Noise_Model_Heron_QPU.pkl")
noise_model = pickle.load(open(file_path, "rb"))

def install_mpi4py():
    """
    Helper function to install mpi4py
    """
    subprocess.check_call(["conda", "install", "-y", "-n", "NoisyCircuits", "-c", "conda-forge", "mpi4py", "openmpi"])

def uninstall_mpi4py():
    """
    Helper function to uninstall mpi4py
    """
    subprocess.check_call(["conda", "uninstall", "-y", "-n", "NoisyCircuits", "-c", "conda-forge", "mpi4py", "openmpi"])

def test_num_qubits_type():
    """
    Test that num_qubits parameter raises a TypeError for non-integer types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=[9])
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits="3")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=(3,))

def test_num_qubits_value():
    """
    Test that num_qubits parameter raises a ValueError for non-positive integers
    """
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=0, noise_model=noise_model)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=-1, noise_model=noise_model)

def test_noise_model_type():
    """
    Test that noise_model parameter raises a TypeError for invalid data types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model="model")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=(noise_model,))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=[noise_model])

def test_backend_qpu_type():
    """
    Test that backend_qpu_type raises a TypeError for invalid data types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, backend_qpu_type=("not", "string"))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, backend_qpu_type=1)
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, backend_qpu_type={})

def test_backend_qpu_value():
    """
    Test that backend_qpu_type raises a ValueError for invalid names
    """
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, backend_qpu_type="octopus")

def test_num_nodes_type():
    """
    Test that num_nodes raises a TypeError for invalid data types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, num_nodes="2")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, num_nodes=(2,))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, num_nodes=[2])

def test_num_nodes_value():
    """
    Test that num_nodes raises a ValueError for invalid values
    """ 
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, num_nodes=0)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, num_nodes=-2)

def test_cores_per_node_type():
    """
    Test that cores_per_node raises a TypeError for invalid data types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_node="1")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_node=(1,))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_node=["1"])

def test_cores_per_node_value():
    """
    Test that cores_per_node raises a ValueError for invalid values
    """
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_node=0)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_node=-1)

def test_cores_per_trajectory_type():
    """
    Test that cores_per_trajectory raises a TypeError for invalid data types.
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_trajectory="1")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_trajectory=(1,))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_trajectory=[1])

def test_cores_per_trajectory_value():
    """
    Test that cores_per_trajectory raises a ValueError for invalid values
    """
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_trajectory=0)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, cores_per_trajectory=-2)

def test_threshold_type():
    """
    Test that threshold raises a TypeError for invalid data types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, threshold="1e-8")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, threshold=(0.1, ))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, threshold=[1e-2])

def test_threshold_value():
    """
    Test that threshold raises a ValueError for invalid values
    """
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, threshold=-2.0)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, threshold=2.0)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, threshold=-1e-1)

def test_verbose_value():
    """
    Test that verbose raises a TypeError for invalid data types
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, verbose="True")
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, verbose=[True])
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=2, noise_model=noise_model, verbose=(True,))

def test_core_count():
    """
    Test that a ValueError is raised when cores_per_trajectory is greater than cores_per_node
    """
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, cores_per_node=1, cores_per_trajectory=2)
    with pytest.raises(ValueError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, cores_per_node=4, cores_per_trajectory=5)

def test_basis_gates_type():
    """
    Test that custom basis_gates entry raises a TypeError when it is not a list of list of strings
    """
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, basis_gates=("something", "else"))
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, basis_gates=["some", "gates"])
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, basis_gates=["something", ["else"]])
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, basis_gates=[[123, 123], [123]])
    with pytest.raises(TypeError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, basis_gates=[[123, 123], ["cz"]])

def test_module_load_error(monkeypatch):
    """
    Test that an ImportError is raised when mpi4py is not installed
    """
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    with pytest.raises(ImportError):
        QuantumCircuitMPI(num_qubits=3, noise_model=noise_model)

def test_execute_qubits_type():
    """
    Test that a TypeError is raised when qubits is not a list of integers
    """
    install_mpi4py()
    circuit = QuantumCircuitMPI(num_qubits=3, noise_model=noise_model)
    with pytest.raises(TypeError):
        circuit.execute(qubits=range(3))
    with pytest.raises(TypeError):
        circuit.execute(qubits=["1", "False", "True"])
    with pytest.raises(TypeError):
        circuit.execute(qubits=[1.2, 1.4, 2.1])
    with pytest.raises(TypeError):
        circuit.execute(qubits=3)

def test_execute_qubits_value():
    """
    Test that a ValueError is raised when qubits contains invalid qubit indices
    """
    circuit = QuantumCircuitMPI(num_qubits=3, noise_model=noise_model)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[-2, -1])
    with pytest.raises(ValueError):
        circuit.execute(qubits=list(range(5)))

def test_execute_num_trajectories_type():
    """
    Test that a TypeError is raised when num_trajectories is not an integer
    """
    circuit = QuantumCircuitMPI(num_qubits=3, noise_model=noise_model)
    with pytest.raises(TypeError):
        circuit.execute(num_trajectories="4")
    with pytest.raises(TypeError):
        circuit.execute(num_trajectories=[4])
    with pytest.raises(TypeError):
        circuit.execute(num_trajectories=(4,))

def test_execute_num_trajectories_value():
    """
    Test that a ValueError is raised when num_trajectories is not a valid number
    """
    circuit = QuantumCircuitMPI(num_qubits=3, noise_model=noise_model)
    with pytest.raises(ValueError):
        circuit.execute(num_trajectories=-10)
    with pytest.raises(ValueError):
        circuit.execute(num_trajectories=0)

def test_execute_runtime_error():
    """
    Test that a RuntimeError is raised when the required MPI ranks and avialable MPI ranks are different
    """
    circuit = QuantumCircuitMPI(num_qubits=3, noise_model=noise_model, cores_per_node=20, cores_per_trajectory=2)
    with pytest.raises(RuntimeError):
        circuit.execute()
    
def test_execute_empty_circuit():
    """
    Test that a ValueError is raised when there are no circuit instructions
    """
    circuit = QuantumCircuitMPI(num_qubits=3, noise_model=noise_model)
    with pytest.raises(ValueError):
        circuit.execute()
    uninstall_mpi4py()