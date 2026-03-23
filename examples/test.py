# import os
# os.environ["OMP_NUM_THREADS"] = "8"  
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"

import time
import pennylane as qml
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

import run
import run_gpu

assert run_gpu.get_gpu_device_count() > 0, "No GPU device found. Please check your CUDA installation."

q = 25
depth = 20
out_cpu = np.zeros(2**q, dtype=np.complex128)
out_gpu = np.zeros(2**q, dtype=np.complex128)
angles = np.random.uniform(-2*np.pi, 2*np.pi, size=(q*depth,))
dev = qml.device("lightning.qubit", wires=q)

def run_pennylane(num_qubits, angles):
    @qml.qnode(dev)
    def circuit(angles, num_qubits):
        for q in range(num_qubits):
            qml.Hadamard(wires=q)
        for d in range(len(angles) // num_qubits):
            for q in range(num_qubits):
                qml.RX(angles[num_qubits*d + q], wires=q)
            for q in range(num_qubits - 1):
                qml.CZ(wires=[q, q+1])
        return qml.state()
    return circuit(angles, num_qubits)

t0 = time.perf_counter_ns()
s_pennylane = run_pennylane(q, angles)
t1 = time.perf_counter_ns()
print(f"Pennylane execution time: {(t1 - t0) / 1e9} seconds")
t0 = time.perf_counter_ns()
run.run_circuit_inplace(angles, q, out_cpu)
t1 = time.perf_counter_ns()
print(f"C++ execution time: {(t1 - t0) / 1e9} seconds")
t0 = time.perf_counter_ns()
run_gpu.run_circuit_gpu(angles, q, out_gpu)
t1 = time.perf_counter_ns()
print(f"GPU execution time: {(t1 - t0) / 1e9} seconds")