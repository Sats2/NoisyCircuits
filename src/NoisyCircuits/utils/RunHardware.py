from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ECRGate as ecr
from qiskit.transpiler import generate_preset_pass_manager
import NoisyCircuits
import numpy as np


class RunOnHardware:
    def __init__(self,
                 token:str=None,
                 backend:str=None,
                 shots:int=None):
        """
        Initialize the RunOnHardware class.

        Args:
            token (str): IBM Quantum token.
            backend (str): Backend name.
            shots (int): Number of shots.
        
        Raises:
            ValueError: When the token, backend or shots are not provided.
            TypeError: When the token, backend or shots are not of the expected type -> (str, str, int) respectively.
            ValueError: When the shots are less than or equal to zero.
            ValueError: When the backend is not available.
        """
        if token is None:
            raise ValueError("Please provide your IBM Quantum token.")
        if backend is None:
            raise ValueError("Please provide a backend name.")
        if shots is None:
            raise ValueError("Please provide the number of shots.")
        if not isinstance(token, str):
            raise TypeError("The token must be a string.")
        if not isinstance(backend, str):
            raise TypeError("The backend name must be a string.")
        if not isinstance(shots, int):
            raise TypeError("The number of shots must be an integer.")
        if shots <= 0:
            raise ValueError("The number of shots must be a positive integer.")
        self.backend = backend
        self.shots = shots
        self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        if self.backend not in self.service.backends():
            raise ValueError(f"The backend '{self.backend}' is not available. Please check your IBM Quantum account.")
        self.circuit_list = []
        self.qubit_list_per_circuit = []
    
    def create_circuits(self,
                       circuit:NoisyCircuits.QuantumCircuit=None,
                       measure_qubits:list[int]=None)->None:
        """
        Method that generates the fully decomposed circuit list in qiskit for IBM Hardware execution.

        Args:
            circuit (NoisyCircuit.QuantumCircuit): The quantum circuit to add in qiskit version.
            measure_qubits (list[int]): The list of qubits to measure.
        
        Raises:
            ValueError: When the circuit is not provided.
            TypeError: When the circuit is not of the expected type -> NoisyCircuits.QuantumCircuit.
            TypeError: When the measure_qubits is not of the expected type -> list.
            TypeError: When the elements of measure_qubits are not of the expected type -> int.
            ValueError: When the circuit is empty.
            ValueError: When the circuit contains gates that are not in the backend's basis gates.
        """
        if circuit is None:
            raise ValueError("Please provide a QuantumCircuit object.")
        if not isinstance(circuit, NoisyCircuits.QuantumCircuit):
            raise TypeError("The circuit must be a QuantumCircuit object.")
        if not isinstance(measure_qubits, list):
            raise TypeError("The measure_qubits must be a list of integers.")
        if not all(isinstance(q, int) for q in measure_qubits):
            raise TypeError("All elements in measure_qubits must be integers.")
        instructions = circuit.instruction_list
        if len(instructions) == 0:
            raise ValueError("The circuit is empty.")
        qc = QuantumCircuit(circuit.num_qubits, len(measure_qubits))
        self.qubit_list_per_circuit.append(list(range(circuit.num_qubits)))
        for inst in instructions:
            gate, qubits, params = inst
            if gate == "x":
                qc.x(qubits[0])
            elif gate == "sx":
                qc.sx(qubits[0])
            elif gate == "rz":
                qc.rz(params[0], qubits[0])
            elif gate == "ecr":
                qc.append(ecr(), qubits)
            else:
                raise ValueError(f"Unsupported gate: {gate} for backend {self.backend}")
        qc.measure(measure_qubits, measure_qubits)
        self.circuit_list.append(qc)
    
    def setup_circuits(self)->None:
        """
        Method that generates the PUB for the circuit list for execution on the IBM Hardware.
        """
        pm = generate_preset_pass_manager(
            backend=self.service.backend(self.backend),
            optimization_level=0,
            initial_layout=self.qubit_list_per_circuit
        )
        self.isa_circuits = pm.run(self.circuit_list)
    
    def run(self)->str:
        """
        Method that submits the generated PUB for execution on the IBM Hardware.

        Raises:
            ValueError: When the maximum circuit execution limit of 10,000,000 is exceeded.

        Returns:
            job_id (str): Returns the Job Id for the submitted batch of PUBs for retrieval.
        """
        sampler = Sampler(mode=self.service.backend(self.backend))
        if len(self.isa_circuits) * self.shots > 10_000_000:
            raise ValueError(f"Maximum circuit executions exceeded. Current setup is {self.shots * len(self.isa_circuits)}/{10_000_000} executions.")
        job = sampler.run(self.isa_circuits, shots=self.shots)
        self.job_id = job.job_id()
        print(f"Job ID: {self.job_id}")
        return self.job_id

    def get_results(self,
                    job_id:str=None)->list[np.ndarray]:
        """
        Method that retrieves the results of the submitted job.

        Args:
            job_id (str, optional): The Job Id for the submitted batch of PUBs. If not provided, uses the last submitted job_id provided the class object is not destroyed.

        Raises:
            TypeError: When the job_id is not of the expected type -> str.
            ValueError: When the job_id is not provided and the class object is destroyed.
        
        Returns:
            list[np.ndarray]: The results of the submitted job. The list contains probabilities of the quantum circuit and the ordering is as inputted into the PUBs.
        """
        if job_id is not None:
            if not isinstance(job_id, str):
                raise TypeError("The job_id must be a string.")
            self.job_id = job_id
        if job_id is None and not hasattr(self, 'job_id'):
            raise ValueError("Please provide a job_id as the class object does not hold any job id.")
        job = self.service.job(self.job_id)
        result = job.result()
        counts_list = [pub.join_data().get_counts() for pub in result]
        result_array_list = []
        for counts_dict in counts_list:
            result_array = np.zeros(len(counts_dict.keys()), dtype=float)
            for key in counts_dict.keys():
                location = int(str(key), 2)
                result_array[location] = counts_dict[key] / self.shots
            result_array_list.append(result_array)
        return result_array_list