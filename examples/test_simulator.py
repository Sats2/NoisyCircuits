from NoisyCircuits.utils import CreateNoiseModel
from NoisyCircuits import QuantumCircuit
import numpy as np
import random
import sys

file_name = "../noise_models/Sample_Noise_Model_Heron_QPU.csv"
noise_model = CreateNoiseModel(file_name, [["x", "sx", "rz", "rx"], ["cz", "rzz"]]).create_noise_model()

circuit = QuantumCircuit(
    num_qubits = 15,
    noise_model = noise_model,
    backend_qpu_type = "heron",
    sim_backend = "custom",
    threshold = 1e-6,
    verbose = False
)

del noise_model


def build_random_circuit(circuit, depth):
    single_gates = {
        "x": lambda p, q: circuit.X(q),
        "sx": lambda p, q: circuit.SX(q),
        "rz": lambda p, q: circuit.RZ(p, q),
        "rx": lambda p, q: circuit.RX(p, q)
    }
    two_gates = {
        "cz": lambda p, q1, q2: circuit.CZ(q1, q2),
        "rzz": lambda p, q1, q2: circuit.RZZ(p, q1, q2)
    }
    for _ in range(depth):
        for i in range(circuit.num_qubits):
            gate_name = random.choice(list(single_gates.keys()))
            single_gates[gate_name](random.uniform(-2*3.14159, 2*3.14159), i)
        for q in range(circuit.num_qubits - 1):
            gate_name = random.choice(list(two_gates.keys()))
            two_gates[gate_name](random.uniform(-2*3.14159, 2*3.14159), q, q + 1)


circuit.refresh()
build_random_circuit(circuit, depth=100)

def density_matrix_simulation(circuit):
    p = circuit.run_with_density_matrix(list(range(circuit.num_qubits)), 50)
    return p

def custom_simulation(circuit):
    p = circuit.execute(list(range(circuit.num_qubits)), 1000, 50)
    return p

def do_nothing():
    pass

arg = sys.argv[1]

if arg == "dm":
    p = density_matrix_simulation(circuit)
elif arg == "custom":
    p = custom_simulation(circuit)
else:
    do_nothing()