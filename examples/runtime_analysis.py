import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from numba import njit
import subprocess
import timeit
import run as simulator


def run_circuit_pennylane_default(angles, num_qubits):
    @qml.qnode(qml.device("default.qubit", wires=num_qubits))
    def circuit(angles, num_qubits):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for d in range(len(angles) // num_qubits):
            for i in range(num_qubits):
                qml.RX(angles[num_qubits*d + i], wires=i)
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i+1])
        return qml.state()
    return circuit(angles, num_qubits)

def run_circuit_pennylane_lightning(angles, num_qubits):
    @qml.qnode(qml.device("lightning.qubit", wires=num_qubits))
    def circuit(angles, num_qubits):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for d in range(len(angles) // num_qubits):
            for i in range(num_qubits):
                qml.RX(angles[num_qubits*d + i], wires=i)
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i+1])
        return qml.state()
    return circuit(angles, num_qubits)

def RX_gate(theta):
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    return np.array([[cosine, -1j*sine], [-1j*sine, cosine]])

def CZ_gate():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def H_gate():
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

@njit
def update_state(U, state, qubit):
    new_state = np.zeros_like(state)
    for g in range(0, len(state), 2**(qubit+1)):
        for i in range(g, 2**(qubit)+g, 1):
            new_state[i] = U[0, 0] * state[i] + U[0, 1] * state[i + 2**qubit]
            new_state[i + 2**qubit] = U[1, 0] * state[i] + U[1, 1] * state[i + 2**qubit]
    return new_state

@njit
def update_state_2q(U, state, qubits):
    new_state = np.zeros_like(state)
    q1, q2 = qubits
    m1 = (1 << q1) - 1
    m2 = (1 << q2) - 1
    iters_tot = len(state) >> 2
    for i in range(iters_tot):
        i_s1 = (i & m1) | ((i & ~m1) << 1)
        pos_00 = (i_s1 & m2) | ((i_s1 & ~m2) << 1)
        pos_01 = pos_00 + (1 << q1)
        pos_10 = pos_00 + (1 << q2)
        pos_11 = pos_00 + (1 << q1) + (1 << q2)
        new_state[pos_00] += U[0,0]*state[pos_00] + U[0,1]*state[pos_01] + U[0,2]*state[pos_10] + U[0,3]*state[pos_11] 
        new_state[pos_01] += U[1,0]*state[pos_00] + U[1,1]*state[pos_01] + U[1,2]*state[pos_10] + U[1,3]*state[pos_11] 
        new_state[pos_10] += U[2,0]*state[pos_00] + U[2,1]*state[pos_01] + U[2,2]*state[pos_10] + U[2,3]*state[pos_11] 
        new_state[pos_11] += U[3,0]*state[pos_00] + U[3,1]*state[pos_01] + U[3,2]*state[pos_10] + U[3,3]*state[pos_11]
    return new_state

def run_circuit_bit_manipulation(angles, num_qubits):
    def apply_circuit(angles, num_qubits):
        state = np.zeros(2**num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        for i in range(num_qubits):
            state = update_state(H_gate(), state, i)
        for d in range(len(angles) // num_qubits):
            for i in range(num_qubits):
                state = update_state(RX_gate(angles[num_qubits*d + i]), state, i)
            for i in range(num_qubits - 1):
                state = update_state_2q(CZ_gate(), state, [i, i+1])
        return state
    return apply_circuit(angles, num_qubits)

def run_circuit_cpp(num_qubits, angles_list):
    return simulator.run_circuit_new(angles_list, num_qubits)

def test_runtime(max_qubits, title_inclusion, title_name, depth=2, log=None):
    qubit_list = []
    mean_pennylane_default = []
    std_pennylane_default = []
    mean_bitmanip = []
    std_bitmanip = []
    mean_pennylane_lightning = []
    std_pennylane_lightning = []
    mean_cpp = []
    std_cpp = []
    num = 5
    for num_qubits in range(1, max_qubits + 1, 1):
        angles = np.random.uniform(-2*np.pi, 2*np.pi, size=(depth * num_qubits,))
        ns = {**globals(), 'num_qubits': num_qubits, 'angles': angles, "depth":depth}
        times_pennylane_default = timeit.repeat("run_circuit_pennylane_default(angles, num_qubits)", 
                                        globals=ns,
                                        repeat=num, 
                                        number=num)
        times_pennylane_lightning = timeit.repeat("run_circuit_pennylane_lightning(angles, num_qubits)", 
                                        globals=ns,
                                        repeat=num, 
                                        number=num)
        times_bitmanip = timeit.repeat("run_circuit_bit_manipulation(angles, num_qubits)", 
                                       globals=ns,
                                       repeat=num,
                                       number=num)
        times_cpp = timeit.repeat("run_circuit_cpp(num_qubits, angles)",
                                        globals=ns,
                                        repeat=num,
                                        number=num)
        mean_pennylane_default.append(np.mean(times_pennylane_default) / num)
        std_pennylane_default.append(np.std(times_pennylane_default) / num)
        mean_bitmanip.append(np.mean(times_bitmanip) / num)
        std_bitmanip.append(np.std(times_bitmanip) / num)
        mean_pennylane_lightning.append(np.mean(times_pennylane_lightning) / num)
        std_pennylane_lightning.append(np.std(times_pennylane_lightning) / num)
        mean_cpp.append(np.mean(times_cpp) / num)
        std_cpp.append(np.std(times_cpp) / num)
        qubit_list.append(num_qubits)
        print(f"Completed {num_qubits} qubits.")
        log.write(f"Completed {num_qubits} qubits.\n")
        print(f"Pennylane (default.qubit): {mean_pennylane_default[-1]} +/- {std_pennylane_default[-1]} seconds")
        log.write(f"Pennylane (default.qubit): {mean_pennylane_default[-1]} +/- {std_pennylane_default[-1]} seconds\n")
        print(f"Pennylane (lightning.qubit): {mean_pennylane_lightning[-1]} +/- {std_pennylane_lightning[-1]} seconds")
        log.write(f"Pennylane (lightning.qubit): {mean_pennylane_lightning[-1]} +/- {std_pennylane_lightning[-1]} seconds\n")
        print(f"Bit Manipulation: {mean_bitmanip[-1]} +/- {std_bitmanip[-1]} seconds")
        log.write(f"Bit Manipulation: {mean_bitmanip[-1]} +/- {std_bitmanip[-1]} seconds\n")
        print(f"C++: {mean_cpp[-1]} +/- {std_cpp[-1]} seconds")
        log.write(f"C++: {mean_cpp[-1]} +/- {std_cpp[-1]} seconds\n")
        print("-" * 40)
        log.write("-" * 40 + "\n")
        log.flush()
    fig = plt.figure(figsize=(10, 6))
    for mean, std, label in zip([mean_pennylane_default, mean_pennylane_lightning, mean_bitmanip, mean_cpp], [std_pennylane_default, std_pennylane_lightning, std_bitmanip, std_cpp], ["Pennylane", "Pennylane Lightning", "Bit Manipulation", "C++"]):
        plt.plot(qubit_list, mean, label=label)
        plt.fill_between(qubit_list, 
                         np.array(mean) - 2*np.array(std),
                         np.array(mean) + 2*np.array(std),
                         alpha=0.3)
    plt.yscale("log")
    plt.xlabel("Number of Qubits")
    plt.xticks(qubit_list)
    plt.ylabel("Average Runtime (seconds)")
    if title_inclusion:
        plt.title(f"Runtime Comparison - (Depth={depth}) - {title_inclusion}")
    else:
        plt.title(f"Runtime Comparison (Depth={depth})")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(title_name + ".png", dpi=500)
    plt.show()

if __name__ == "__main__":
    max_qubits = 25
    log = open("Runtime_Analysis_with_Cpp_Module_modified.log", "w")
    q = np.random.randint(15, 25)
    angles = np.random.uniform(-2*np.pi, 2*np.pi, size=(q*20,))
    p_pennylane = run_circuit_pennylane_default(angles, q)
    p_lightning = run_circuit_pennylane_lightning(angles, q)
    p_bit = run_circuit_bit_manipulation(angles, q)
    p_cpp = simulator.run_circuit_new(angles, q)
    p_cpp = np.array(p_cpp)
    p_cpp = p_cpp.reshape([2]*q).transpose(list(range(q))[::-1]).reshape(-1)
    p_bit = p_bit.reshape([2]*q).transpose(list(range(q))[::-1]).reshape(-1)
    fid_default = np.vdot(p_pennylane, p_bit).real ** 2
    fid_lightning = np.vdot(p_lightning, p_bit).real ** 2
    fig_cpp = np.vdot(p_pennylane, p_cpp).real ** 2
    del p_pennylane, p_lightning, p_bit, p_cpp
    del angles
    print(f"Used {q} qubits for fidelity test.")
    log.write(f"Used {q} qubits for fidelity test.\n")
    print(f"Fidelity with Pennylane default.qubit: {fid_default:.6f}")
    log.write(f"Fidelity with Pennylane default.qubit: {fid_default:.6f}\n")
    print(f"Fidelity with Pennylane lightning.qubit: {fid_lightning:.6f}")
    log.write(f"Fidelity with Pennylane lightning.qubit: {fid_lightning:.6f}\n")
    print(f"Fidelity of C++ implementation against Pennylane: {fig_cpp:.6f}")
    log.write(f"Fidelity of C++ implementation against Pennylane: {fig_cpp:.6f}\n")
    print("----"*20 + "\n")
    log.write("----"*20 + "\n")
    assert np.isclose(fid_default, 1.0, atol=1e-6), "Fidelity with Pennylane default.qubit is not close to 1!"
    assert np.isclose(fid_lightning, 1.0, atol=1e-6), "Fidelity with Pennylane lightning.qubit is not close to 1!"
    test_runtime(max_qubits, title_inclusion="", title_name="Runtime_comparison_with_CPP_module", depth=20, log=log)
    log.close()