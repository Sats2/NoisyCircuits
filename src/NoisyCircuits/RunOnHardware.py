"""
This module contains the `RunOnHardware` class which allows users to run quantum circuits on IBM Quantum Hardware. The class provides methods to create circuits (according to hardware requirements), set them up for execution, submit them to the hardware, check job status, cancel jobs, and retrieve results. IBM Quantum's Qiskit Runtime Service is utilized for backend communication and job management. The retrieved results are formatted as probability distributions from little Endian format to big Endian format for easy analysis.

Example usage:
    >>> from NoisyCircuits.RunOnHardware import RunOnHardware
    >>> from NoisyCircuits.QuantumCircuit import QuantumCircuit
    >>> circuit = QuantumCircuit(num_qubits=2, noise_model=noise_model, num_cores=2, num_trajectories=100)
    >>> circuit.H(qubit=0)
    >>> circuit.CX(control=0, target=1)
    >>> runner = RunOnHardware(token=your_token, backend="ibm_fez", shots=1024)
    >>> runner.create_circuits(circuit=circuit, measure_qubits=[0, 1])
    >>> runner.setup_circuits()
    >>> job_id = runner.run()
    >>> status = runner.status(job_id=job_id)
    'DONE'
    >>> results = runner.get_results(job_id=job_id)
    >>> circuit.execute(qubits=[0, 1])
    [0.40313749, 0.09695667, 0.09692287, 0.40298296]
    >>> print(results)
    [0.40813749, 0.09195667, 0.10892288, 0.39098296]

It should be noted that the IBM Quantum backends have a maximum limit of 10,000,000 circuit executions per job submission. Users should ensure that the product of the number of circuits and shots does not exceed this limit to avoid job submission errors.
"""
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ECRGate as ecr
from qiskit.transpiler import generate_preset_pass_manager
from NoisyCircuits.QuantumCircuit import QuantumCircuit as NoisyCircuitsQuantumCircuit
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
        backend_list = [backend.name for backend in self.service.backends()]
        if self.backend not in backend_list:
            raise ValueError(f"The backend '{self.backend}' is not available. Please check your IBM Quantum account.")
        self.circuit_list = []
        self.qubit_list_per_circuit = []
    
    def create_circuits(self,
                       circuit:NoisyCircuitsQuantumCircuit=None,
                       measure_qubits:list[int]=None)->None:
        """
        Method that generates the fully decomposed circuit list in qiskit for IBM Hardware execution.

        Args:
            circuit (NoisyCircuit.QuantumCircuit.QuantumCircuit): The quantum circuit to add in qiskit version.
            measure_qubits (list[int]): The list of qubits to measure.
        
        Raises:
            ValueError: When the circuit is not provided.
            TypeError: When the circuit is not of the expected type -> NoisyCircuits.QuantumCircuit.
            TypeError: When the measure_qubits is not of the expected type -> list.
            TypeError: When the elements of measure_qubits are not of the expected type -> int.
            ValueError: When the circuit is empty.
            ValueError: When the measure_qubits contain invalid qubit indices.
            ValueError: When the circuit contains gates that are not in the backend's basis gates.
        """
        if circuit is None:
            raise ValueError("Please provide a QuantumCircuit object.")
        if not isinstance(circuit, NoisyCircuitsQuantumCircuit):
            raise TypeError("The circuit must be a QuantumCircuit object.")
        if not isinstance(measure_qubits, list):
            raise TypeError("The measure_qubits must be a list of integers.")
        if not all(isinstance(q, int) for q in measure_qubits):
            raise TypeError("All elements in measure_qubits must be integers.")
        if np.max(measure_qubits) >= circuit.num_qubits or np.min(measure_qubits) < 0:
            raise ValueError("The measure_qubits contain invalid qubit indices.")
        instructions = circuit.instruction_list
        if len(instructions) == 0:
            raise ValueError("The circuit is empty.")
        qc = QuantumCircuit(circuit.num_qubits, len(measure_qubits))
        instruction_map = {
            "x" : lambda q: qc.x(q[0]),
            "sx": lambda q: qc.sx(q[0]),
            "rz": lambda p, q: qc.rz(p, q[0]),
            "rx": lambda p, q: qc.rx(p, q[0]),
            "ecr": lambda q: qc.append(ecr(), [q[0], q[1]]),
            "cz": lambda q: qc.cz(q[0], q[1]),
            "rzz": lambda p, q: qc.rzz(p, q[0], q[1])
        }
        self.qubit_list_per_circuit.append(list(range(circuit.num_qubits)))
        for inst in instructions:
            gate, qubits, params = inst
            try:
                if params is not None:
                    instruction_map[gate](params, qubits)
                else:
                    instruction_map[gate](qubits)
            except KeyError:
                raise ValueError(f"The gate '{gate}' is not supported on the backend '{self.backend}'. Please check the backend's basis gates.")
        qc.measure(measure_qubits, measure_qubits)
        self.circuit_list.append(qc)
    
    def setup_circuits(self)->None:
        """
        Method that generates the PUB for the circuit list for execution on the IBM Hardware.
        """
        self.isa_circuits = []
        for circuit, layout in zip(self.circuit_list, self.qubit_list_per_circuit):
            pass_manager = generate_preset_pass_manager(
                backend = self.service.backend(self.backend),
                optimization_level=0,
                initial_layout=layout
            )
            self.isa_circuits.append(pass_manager.run(circuit))
    
    def run(self)->str:
        """
        Method that submits the generated PUB for execution on the IBM Hardware.

        Raises:
            ValueError: When the maximum circuit execution limit of 10,000,000 is exceeded.

        Returns:
            str: Returns the Job Id for the submitted batch of PUBs for retrieval.
        """
        sampler = Sampler(mode=self.service.backend(self.backend))
        if len(self.isa_circuits) * self.shots > 10_000_000:
            raise ValueError(f"Maximum circuit executions exceeded. Current setup is {self.shots * len(self.isa_circuits)}/{10_000_000} executions.")
        job = sampler.run(self.isa_circuits, shots=self.shots)
        self.job_id = job.job_id()
        print(f"Job ID: {self.job_id}")
        return self.job_id
    
    def status(self,
               job_id:str=None)->str:
        """
        Method that retrieves the status of the submitted job.

        Args:
            job_id (str, optional): The Job Id for the submitted batch of PUBs. If not provided, uses the 
                                    last submitted job_id provided the class object is not destroyed.
        
        Raises:
            TypeError: When the job_id is not of the expected type -> str.
            ValueError: When the job_id is not provided and the class object is destroyed.

        Returns:
            str: The status of the submitted job.
        """
        if job_id is not None:
            if not isinstance(job_id, str):
                raise TypeError("The job_id must be a string.")
            self.job_id = job_id
        if job_id is None and not hasattr(self, 'job_id'):
            raise ValueError("Please provide a job_id as the class object does not hold any job id.")
        job = self.service.job(self.job_id)
        return job.status()
    
    def cancel(self,
               job_id:str=None)->None:
        """
        Method to cancel the submitted job.

        Args:
            job_id (str, optional): The Job Id for the submitted batch of PUBs. If not provided, uses the 
                                    last submitted job_id provided the class object is not destroyed.

        Raises:
            TypeError: When the job_id is not of the expected type -> str.
            ValueError: When the job_id is not provided and the class object is destroyed.
        """
        if job_id is not None:
            if not isinstance(job_id, str):
                raise TypeError("The job_id must be a string.")
            self.job_id = job_id
        if job_id is None and not hasattr(self, 'job_id'):
            raise ValueError("Please provide a job_id as the class object does not hold any job id.")
        job = self.service.job(self.job_id)
        try:
            job.cancel()
        except Exception as e:
            print(f"An error occurred while cancelling the job: {e}")

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
            num_bits = 2**len(next(iter(counts_dict)))
            result_array = np.zeros(num_bits, dtype=float)
            for key in counts_dict.keys():
                location = str(key)[::-1]
                location = int(location, 2)
                result_array[location]  = counts_dict[key] / self.shots
            result_array_list.append(result_array)
        return result_array_list