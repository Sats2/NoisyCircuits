from NoisyCircuits.QuantumCircuit import QuantumCircuit
from NoisyCircuits.RunOnHardware import RunOnHardware
import pytest
import pickle
import json
import os
from pathlib import Path

file_path = os.path.join(Path(__file__).parent.parent, "noise_models/Noise_Model_Heron_QPU.pkl")
circuit = QuantumCircuit(
    num_qubits=2,
    noise_model=pickle.load(open(file_path, "rb")),
    backend_qpu_type="heron",
    num_trajectories=100,
    jsonize=True,
    verbose=True
)

def test_token_type():
    """
    Test that the token parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        RunOnHardware(token=12345, backend="ibm_fez", shots=1024)
    with pytest.raises(TypeError):
        RunOnHardware(token=["token_string"], backend="ibm_fez", shots=1024)

def test_backend_type():
    """
    Test that the backend parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        RunOnHardware(token="valid_token", backend=12345, shots=1024)
    with pytest.raises(TypeError):
        RunOnHardware(token="valid_token", backend=["ibm_fez"], shots=1024)

def test_shots_type():
    """
    Test that the shots parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        RunOnHardware(token="valid_token", backend="ibm_fez", shots="1024")
    with pytest.raises(TypeError):
        RunOnHardware(token="valid_token", backend="ibm_fez", shots=1024.5)

def test_empty_token():
    """
    Test that an empty token raises ValueError.
    """
    with pytest.raises(ValueError):
        RunOnHardware(backend="ibm_fez", shots=1024)

def test_empty_backend():
    """
    Test that an empty backend raises ValueError.
    """
    with pytest.raises(ValueError):
        RunOnHardware(token="valid_token", shots=1024)

def test_empty_shots():
    """
    Test that an empty shots raises ValueError.
    """
    with pytest.raises(ValueError):
        RunOnHardware(token="valid_token", backend="ibm_fez")

def test_shots_value():
    """
    Test that invalid shots values raise ValueError.
    """
    with pytest.raises(ValueError):
        RunOnHardware(token="valid_token", backend="ibm_fez", shots=0)
    with pytest.raises(ValueError):
        RunOnHardware(token="valid_token", backend="ibm_fez", shots=-100)

@pytest.mark.localonly
def test_invalid_backend():
    """
    Test that an invalid backend name raises ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    with pytest.raises(ValueError):
        RunOnHardware(token=api_token, backend="ibm_perth", shots=1024)

@pytest.mark.localonly
def test_create_circuits_no_circuit():
    """
    Test that creating circuits without a circuit raises ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(ValueError):
        runner.create_circuits(measure_qubits=[0, 1])

@pytest.mark.localonly
def test_create_circuits_circuit_type():
    """
    Test that invalid circuit types raise TypeError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(TypeError):
        runner.create_circuits(circuit="not_a_circuit", measure_qubits=[0, 1])
    with pytest.raises(TypeError):
        runner.create_circuits(circuit=12345, measure_qubits=[0, 1])

@pytest.mark.localonly    
def test_create_circuits_measure_qubits_type():
    """
    Test that invalid measure_qubits types raise TypeError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(TypeError):
        runner.create_circuits(circuit=circuit, measure_qubits="0,1")
    with pytest.raises(TypeError):
        runner.create_circuits(circuit=circuit, measure_qubits=12345)
    with pytest.raises(TypeError):
        runner.create_circuits(circuit=circuit, measure_qubits=[0, "1", 2])
    with pytest.raises(TypeError):
        runner.create_circuits(circuit=circuit, measure_qubits=[0, 1.5, 2])

@pytest.mark.localonly
def test_create_circuits_empty_circuit():
    """
    Test that creating circuits with an empty circuit raises ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(ValueError):
        runner.create_circuits(circuit=circuit, measure_qubits=[0, 1])

@pytest.mark.localonly
def test_create_circuits_invalid_measure_qubits():
    """
    Test that invalid measure_qubits values raise ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(ValueError):
        runner.create_circuits(circuit=circuit, measure_qubits=[0, 2])
    with pytest.raises(ValueError):
        runner.create_circuits(circuit=circuit, measure_qubits=[-1, 0])

@pytest.mark.localonly
def test_run_max_circuit_eval():
    """
    Test that a ValueError is raised when exceeding maximum circuit evaluation limit (no.of circuits * no. of shots).
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    runner.shots = 10_500_000
    circuit.H(0)
    circuit.CX(0, 1)
    runner.create_circuits(circuit=circuit, measure_qubits=[0, 1])
    runner.setup_circuits()
    with pytest.raises(ValueError):
        runner.run()
    circuit.refresh()
    runner.shots = 1024
    runner.circuit_list = []

@pytest.mark.localonly
def test_status_jobid_type():
    """
    Test that invalid job_id types for status raise TypeError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(TypeError):
        runner.status(job_id=12345)
    with pytest.raises(TypeError):
        runner.status(job_id=["job_id_string"])

@pytest.mark.localonly
def test_cancel_jobid_type():
    """
    Test that invalid job_id types for cancel raise TypeError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(TypeError):
        runner.cancel(job_id=12345)
    with pytest.raises(TypeError):
        runner.cancel(job_id=["job_id_string"])

@pytest.mark.localonly
def test_cancel_no_jobid():
    """
    Test that cancel without a job_id raises ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(ValueError):
        runner.cancel()

@pytest.mark.localonly
def test_status_no_jobid():
    """
    Test that status without a job_id raises ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(ValueError):
        runner.status()

@pytest.mark.localonly
def test_get_results_jobid_type():
    """
    Test that invalid job_id types for get_results raise TypeError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(TypeError):
        runner.get_results(job_id=12345)
    with pytest.raises(TypeError):
        runner.get_results(job_id=["job_id_string"])

@pytest.mark.localonly
def test_get_results_no_jobid():
    """
    Test that get_results without a job_id raises ValueError.
    """
    api_token = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]
    runner = RunOnHardware(token=api_token, backend="ibm_fez", shots=1024)
    with pytest.raises(ValueError):
        runner.get_results()

circuit.shutdown()