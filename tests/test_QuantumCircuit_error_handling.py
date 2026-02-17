from NoisyCircuits.QuantumCircuit import QuantumCircuit
import pytest
import pickle
import os
from pathlib import Path


file_path = os.path.join(Path(__file__).parent.parent, "noise_models/Noise_Model_Heron_QPU.pkl")

def test_num_qubits_type():
    """
    Test that num_qubits parameter raises TypeError for non-integer types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits="5", noise_model=noise_model, backend_qpu_type="heron", num_trajectories=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=5.5, noise_model=noise_model, backend_qpu_type="heron", num_trajectories=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=[5], noise_model=noise_model, backend_qpu_type="heron", num_trajectories=1)

def test_num_qubits_value():
    """
    Test that num_qubits parameter raises ValueError for invalid values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=-2, noise_model=noise_model, backend_qpu_type="heron", num_trajectories=1)

def test_noise_model_type():
    """
    Test that noise_model parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model="invalid_model", backend_qpu_type="heron", num_trajectories=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=123, backend_qpu_type="heron", num_trajectories=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=[152, "test"], backend_qpu_type="heron", num_trajectories=1)

def test_backend_qpu_type():
    """
    Test that backend_qpu_type parameter raises TypeError for non-string types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type=123, num_trajectories=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type=["type"], num_trajectories=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type={"type": "Heron"}, num_trajectories=1)

def test_backend_qpu_value():
    """
    Test that backend_qpu_type parameter raises ValueError for invalid string values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="InvalidQPU", num_trajectories=1)
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="", num_trajectories=1)

def test_num_trajectory_type():
    """
    Test that num_trajectories parameter raises TypeError for non-integer types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", num_trajectories="100")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", num_trajectories=50.5)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", num_trajectories=[100])

def test_num_trajectory_value():
    """
    Test that num_trajectories parameter raises ValueError for invalid values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", num_trajectories=-10)
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", num_trajectories=0)

def test_threshold_type():
    """
    Test that threshold parameter raises TypeError for non-float types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", 
                       num_trajectories=10, threshold="0.01")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", 
                       num_trajectories=10, threshold=[0.01])
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold={"threshold": 0.01})
        
def test_threshold_value():
    """
    Test that threshold parameter raises ValueError for out-of-bounds values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=-0.1)
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1.5)

def test_num_cores_type():
    """
    Test that num_cores parameter raises TypeError for non-integer types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores="4")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=2.5)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=[4])

def test_num_cores_value():
    """
    Test that num_cores parameter raises ValueError for invalid values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=-2)
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=0)

def test_sim_backend_type():
    """
    Test that sim_backend parameter raises TypeError for non-string types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, sim_backend=123)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, sim_backend=["pennylane"])
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, sim_backend={"backend": "pennylane"})
        
def test_sim_backend_value():
    """
    Test that sim_backend parameter raises ValueError for invalid string values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, sim_backend="invalid_backend")
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, sim_backend="")

def test_sim_backend_default():
    """
    Test that sim_backend parameter defaults to "pennylane" when not provided.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4)
    assert circuit.sim_backend == "pennylane"
    circuit.shutdown()

def test_jsonize_type():
    """
    Test that jsonize parameter raises TypeError for non-boolean types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize="True")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=1)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=[True])

def test_verbose_type():
    """
    Test that verbose parameter raises TypeError for non-boolean types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose="False")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=0)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=[False])
        
def test_execute_qubits_type():
    """
    Test that execute method raises TypeError for invalid qubits parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(TypeError):
        circuit.execute(qubits="0,1")
    with pytest.raises(TypeError):
        circuit.execute(qubits=0)
    with pytest.raises(TypeError):
        circuit.execute(qubits={0,1})
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0,"1"])
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0,1.5])
    circuit.shutdown()

def test_execute_qubits_value():
    """
    Test that execute method raises ValueError for out-of-bounds qubits.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[-1, 0])
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0, 2])
    circuit.shutdown()

def test_execute_num_trajectories_type():
    """
    Test that execute method raises TypeError for invalid num_trajectories parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0,1], num_trajectories="5")
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0,1], num_trajectories=3.5)
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0,1], num_trajectories=[5])
    circuit.shutdown()

def test_execute_num_trajectories_value():
    """
    Test that execute method raises ValueError for invalid num_trajectories parameter values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0,1], num_trajectories=-5)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0,1], num_trajectories=0)
    circuit.shutdown()

def test_execute_empty_instructions():
    """
    Test that execute method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0,1], num_trajectories=5)
    circuit.shutdown()

def test_density_matrix_empty_instructions():
    """
    Test that run_with_density_matrix method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[0,1])
    circuit.shutdown()

def test_density_matrix_qubits_type():
    """
    Test that run_with_density_matrix method raises TypeError for invalid qubits parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits="0,1")
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits=0)
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits={0,1})
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits=[0,"1"])
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits=[0,1.5])
    circuit.shutdown()

def test_density_matrix_qubits_value():
    """
    Test that run_with_density_matrix method raises ValueError for out-of-bounds qubits.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[-1, 0])
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[0, 2])
    circuit.shutdown()

def test_pure_state_empty_instructions():
    """
    Test that run_pure_state method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[0,1])
    circuit.shutdown()

def test_pure_state_qubits_type():
    """
    Test that run_pure_state method raises TypeError for invalid qubits parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits="0,1")
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=0)
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits={0,1})
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0,"1"])
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0,1.5])
    circuit.shutdown()

def test_pure_state_qubits_value():
    """
    Test that run_pure_state method raises ValueError for out-of-bounds qubits.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[-1, 0])
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[0, 2])
    circuit.shutdown()

def test_draw_circuit_empty_instructions():
    """
    Test that draw_circuit method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    with pytest.raises(ValueError):
        circuit.draw_circuit()
    circuit.shutdown()

def test_draw_circuit_style_type():
    """
    Test that draw_circuit method raises TypeError for invalid style parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    with pytest.raises(TypeError):
        circuit.draw_circuit(style=123)
    with pytest.raises(TypeError):
        circuit.draw_circuit(style=["default"])
    with pytest.raises(TypeError):
        circuit.draw_circuit(style={"style": "default"})
    circuit.shutdown()

def test_draw_circuit_style_value():
    """
    Test that draw_circuit method raises ValueError for invalid style parameter values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron",
                       num_trajectories=10, threshold=1e-4, num_cores=4, jsonize=True, verbose=False)
    circuit.H(0)
    with pytest.raises(ValueError):
        circuit.draw_circuit(style="invalid_style")
    with pytest.raises(ValueError):
        circuit.draw_circuit(style="")
    circuit.shutdown()

