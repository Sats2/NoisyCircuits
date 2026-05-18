#!/usr/bin/env python
# coding: utf-8

# In[1]:


import simulator
from NoisyCircuits.utils.CreateNoiseModel import CreateNoiseModel
import numpy as np
import pennylane as qml


# In[2]:


noise_model = CreateNoiseModel("../noise_models/Sample_Noise_Model_Heron_QPU.csv", [["sx", "x", "rx", "rz"], ["cz", "rzz"]]).create_noise_model()


# In[3]:


from NoisyCircuits.utils.BuildQubitGateModel import BuildModel


# In[4]:


modeller = BuildModel(
    noise_model,
    10,
    10,
    1e-6,
    [["x", "sx", "rx", "rz"], ["cz", "rzz"]],
    False
)
single, double, measure, connection = modeller.build_qubit_gate_model()


# In[5]:


def build_random_unitary(n):
    N = 1 << n
    random_matrix = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    Q, R = np.linalg.qr(random_matrix)
    D = np.diag(R)
    D = D / np.abs(D)
    return Q @ np.diag(D)


# In[6]:


import random

def generate_random_qubit_list(n, b, a=0):
    return random.sample(range(a, b), n)


# In[7]:


dev = qml.device("lightning.qubit", wires=10)

def test_unitary(U, apply_qubits, dev=dev):
    @qml.qnode(dev)
    def apply_unitary():
        qml.QubitUnitary(U, wires=apply_qubits)
        return qml.state()
    return apply_unitary()


# In[30]:


q = 4
unitary = build_random_unitary(q)
apply_qubits = generate_random_qubit_list(q, 10)
state_pennylane = test_unitary(unitary, apply_qubits)
state_custom = np.zeros(1 << 10, dtype=np.complex128)
instruction_lst = [["unitary", apply_qubits[::-1], unitary]]
state_custom[0] = 1.0
simulator.simulate_circuit(instruction_lst, state_custom, single, double, 10, False, 1, True, 10)
state_custom = state_custom.reshape([2]*10).transpose(list(range(10))[::-1]).reshape(-1)
print("States Match: ", np.allclose(state_custom, state_pennylane, atol=1e-10))

