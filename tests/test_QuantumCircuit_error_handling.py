# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

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
        QuantumCircuit(num_qubits="5", noise_model=noise_model, backend_qpu_type="heron")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=5.5, noise_model=noise_model, backend_qpu_type="heron")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=[5], noise_model=noise_model, backend_qpu_type="heron")

def test_num_qubits_value():
    """
    Test that num_qubits parameter raises ValueError for invalid values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=-2, noise_model=noise_model, backend_qpu_type="heron")

def test_noise_model_type():
    """
    Test that noise_model parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model="invalid_model", backend_qpu_type="heron")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=123, backend_qpu_type="heron")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=[152, "test"], backend_qpu_type="heron")

def test_backend_qpu_type():
    """
    Test that backend_qpu_type parameter raises TypeError for non-string types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type=123)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type=["type"])
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type={"type": "Heron"})

def test_backend_qpu_value():
    """
    Test that backend_qpu_type parameter raises ValueError for invalid string values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="InvalidQPU")
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="")

def test_threshold_type():
    """
    Test that threshold parameter raises TypeError for non-float types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold="0.01")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=[0.01])
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold={"threshold": 0.01})
        
def test_threshold_value():
    """
    Test that threshold parameter raises ValueError for out-of-bounds values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=-0.1)
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1.5)

def test_sim_backend_type():
    """
    Test that sim_backend parameter raises TypeError for non-string types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, sim_backend=123)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, sim_backend=["pennylane"])
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, sim_backend={"backend": "pennylane"})
        
def test_sim_backend_value():
    """
    Test that sim_backend parameter raises ValueError for invalid string values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, sim_backend="invalid_backend")
    with pytest.raises(ValueError):
        QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, sim_backend="")

def test_sim_backend_default():
    """
    Test that sim_backend parameter defaults to "custom" when not provided.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=1, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4)
    assert circuit.sim_backend == "custom"
    circuit.shutdown()

def test_verbose_type():
    """
    Test that verbose parameter raises TypeError for non-boolean types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose="False")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=0)
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=[False])

def test_use_fractional_type():
    """
    Test that use_fractional parameter raises TypeError for non-boolean types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, use_fractional="True")
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, use_fractional=[True])
    with pytest.raises(TypeError):
        QuantumCircuit(num_qubits=2, noise_model=noise_model, use_fractional=1)
        
def test_execute_qubits_type():
    """
    Test that execute method raises TypeError for invalid qubits parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0,1], num_trajectories=5)
    circuit.shutdown()

def test_execute_num_cores_type():
    """
    Test that execute method raises TypeError for invalid num_cores parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0, 1)
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0, 1], num_trajectories=10, num_cores="4")
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0, 1], num_trajectories=10, num_cores=4.5)
    with pytest.raises(TypeError):
        circuit.execute(qubits=[0, 1], num_trajectories=10, num_cores=[4])
    circuit.shutdown()

def test_execute_num_cores_value():
    """
    Test that execute method raises ValueError for invalid num_cores parameter values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0, 1)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0,1], num_trajectories=10, num_cores=-2)
    with pytest.raises(ValueError):
        circuit.execute(qubits=[0,1], num_trajectories=10, num_cores=0)
    circuit.shutdown()

def test_density_matrix_empty_instructions():
    """
    Test that run_with_density_matrix method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[0,1])
    circuit.shutdown()

def test_density_matrix_qubits_type():
    """
    Test that run_with_density_matrix method raises TypeError for invalid qubits parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[-1, 0])
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[0, 2])
    circuit.shutdown()

def test_density_matrix_num_cores_type():
    """
    Test that run_with_density_matrix method raises TypeError for invalid num_cores parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0, 1)
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits=[0, 1], num_cores="4")
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits=[0, 1], num_cores=4.5)
    with pytest.raises(TypeError):
        circuit.run_with_density_matrix(qubits=[0, 1], num_cores=[4])
    circuit.shutdown()

def test_density_matrix_num_cores_value():
    """
    Test that run_with_density_matrix method raises ValueError for invalid num_cores parameter values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0, 1)
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[0,1], num_cores=-2)
    with pytest.raises(ValueError):
        circuit.run_with_density_matrix(qubits=[0,1], num_cores=0)
    circuit.shutdown()

def test_pure_state_empty_instructions():
    """
    Test that run_pure_state method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[0,1])
    circuit.shutdown()

def test_pure_state_qubits_type():
    """
    Test that run_pure_state method raises TypeError for invalid qubits parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[-1, 0])
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[0, 2])
    circuit.shutdown()

def test_pure_state_num_cores_type():
    """
    Test that run_pure_state method raises TypeError for invalid num_cores parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0, 1)
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0, 1], num_cores="4")
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0, 1], num_cores=4.5)
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0, 1], num_cores=[4])
    circuit.shutdown()

def test_pure_state_num_cores_value():
    """
    Test that run_pure_state method raises ValueError for invalid num_cores parameter values.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0, 1)
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[0,1], num_cores=-2)
    with pytest.raises(ValueError):
        circuit.run_pure_state(qubits=[0,1], num_cores=0)
    circuit.shutdown()

def test_pure_state_return_statevector_type():
    """
    Test that run_pure_state method raises TypeError for invalid return_statevector parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    circuit.CZ(0,1)
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0, 1], num_cores=1, return_statevector="True")
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0, 1], num_cores=1, return_statevector=1)
    with pytest.raises(TypeError):
        circuit.run_pure_state(qubits=[0, 1], num_cores=1, return_statevector=[True])
    circuit.shutdown()

def test_draw_circuit_empty_instructions():
    """
    Test that draw_circuit method raises ValueError when there are no instructions in the circuit.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    with pytest.raises(ValueError):
        circuit.draw_circuit()
    circuit.shutdown()

def test_draw_circuit_style_type():
    """
    Test that draw_circuit method raises TypeError for invalid style parameter types.
    """
    noise_model = pickle.load(open(file_path, "rb"))
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
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
    circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, backend_qpu_type="heron", threshold=1e-4, verbose=False)
    circuit.H(0)
    with pytest.raises(ValueError):
        circuit.draw_circuit(style="invalid_style")
    with pytest.raises(ValueError):
        circuit.draw_circuit(style="")
    circuit.shutdown()