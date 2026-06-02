from NoisyCircuits import QuantumCircuit as QC
from NoisyCircuits.utils.CreateNoiseModel import GetNoiseModel, CreateNoiseModel
import pickle
import os
import json
import numpy as np
import pickle


num_qubits = 2
num_cores = 10
threshold = 1e-6
verbose = False
qpu_type = "heron" 
sim_backend = "custom" # Choose between "custom", "qulacs", "pennylane" and "qiskit"

with open("noise_model.pkl", "rb") as f:
    noise_model = pickle.load(f)

nqc = QC(
    num_qubits=num_qubits,
    noise_model=noise_model,
    backend_qpu_type=qpu_type,
    sim_backend=sim_backend,
    threshold=threshold,
    verbose=verbose
)

nqc.refresh()
nqc.RY(theta=1.2, qubit=0)
nqc.RY(theta=0.5, qubit=1)
nqc.SWAP(qubit1=0, qubit2=1)

print("Returning pure state")
print(nqc.run_pure_state(qubits=[1], num_cores=1, return_statevector=False))
print("\n")
print("Returning Density Matrix")
print(nqc.run_with_density_matrix(qubits=[1], num_cores=1))
print("\nReturning MCWF")
print(nqc.execute(qubits=[1], num_trajectories=1000, num_cores=num_cores))