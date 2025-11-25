from NoisyCircuits.utils.GetNoiseModel import GetNoiseModel
from NoisyCircuits.QuantumCircuit import QuantumCircuit
import pytest
import pickle
import os
import json

@pytest.fixture
def quantum_circuit_heron():
    return QuantumCircuit(
        num_qubits=2,
        noise_model=pickle.load(open("../noise_models/Noise_Model_Heron_QPU.pkl", "rb")),
        backend_qpu_type="heron",
        num_trajectories=100,
        jsonize=True,
        verbose=True
    )

@pytest.fixture
def quantum_circuit_eagle():
    return QuantumCircuit(
        num_qubits=2,
        noise_model=pickle.load(open("../noise_models/Noise_Model_Eagle_QPU.pkl", "rb")),
        backend_qpu_type="eagle",
        num_trajectories=100,
        jsonize=True,
        verbose=True
    )

@pytest.fixture
def noise_model_heron():
    return GetNoiseModel(
        backend_name="ibm_fez",
        token=json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    )