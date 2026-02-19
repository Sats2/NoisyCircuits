"""
This module allows the user to create the raw noise model dictionary from user provided calibration data (CSV file) from an actual Quantum Hardware without the requirement for any tokens. This raw noise model can be provided to the QuantumCircuit or BuildQubitGateModel modules for post-processing. The BuildQubitGateModel module returns the post-processed noise operators for each standardized gate on each qubit whereas the QuantumCircuit module uses the post-processed noise model to perform quantum circuit simulations under noise.

Additionally, the user must specify the basis gates for the quantum hardware in order to create the noise model. The basis gates must be provided in a list of lists format where the first list contains the single qubit basis gates and the second list contains the two qubit basis gates. For example, if the single qubit basis gates are "x", "sx" and "rz" and the two qubit basis gate is "cz", then the basis gates should be provided as [["x", "sx", "rz"], ["cz"]].

Example:
    >>> from NoisyCircuits.utils.CreateNoiseModel import CreateNoiseModel
    >>> calibration_file_path = "path/to/calibration_data.csv"
    >>> basis_gates = [["x", "sx", "rz"], ["cz"]]
    >>> noise_model = CreateNoiseModel(calibration_data_file=calibration_file_path, basis_gates=basis_gates).create_noise_model()
"""

from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, readout_error
import pandas as pd
import numpy as np


class CreateNoiseModel:
    """
    A class to create a noise model from user provided calibration data.

    The definition of the header in the CSV file (in table view) should be as follows:

    +-------+---------+---------+-------------+-------------+--------------+--------------+-------------+--------------+
    | Qubit | T1 (us) | T2 (us) | Prob meas 0 | Prob meas 1 | Single Qubit | Single Qubit | Gate Length | Two Qubit    |
    |       |         |         | prep 1      | prep 0      | Gate Length  | Basis Gate   | (ns)        | Basis Gate   |
    |       |         |         |             |             | (ns)         | Error        |             | Error        |
    +=======+=========+=========+===============+=============+=============+==============+==============+=============+

    An example of the content of the CSV file (in table view) where the basis gates are [["x", "sx"], ["cz"]] is as follows:

    +-------+---------+---------+--------------------+--------------------+-------------------------------+----------+----------+------------------+--------------------+
    | Qubit | T1 (us) | T2 (us) | Prob meas 0 prep 1 | Prob meas 1 prep 0 | Single Qubit Gate Length (ns) |  x Error | sx Error | Gate Length (ns) |      cz Error      |
    +=======+=========+=========+====================+====================+===============================+==========+==========+==================+====================+
    |   0   |  50.534 |  19.955 |       0.0789       |       0.1316       |               24              | 0.002118 | 0.002118 |       1:68       |      1:0.0207      |
    +-------+---------+---------+--------------------+--------------------+-------------------------------+----------+----------+------------------+--------------------+
    |   1   | 153.049 | 144.814 |       0.0696       |       0.1243       |               24              |  0.00052 |  0.00052 |     2:68;0:68    | 2:0.00479;0:0.0207 |
    +-------+---------+---------+--------------------+--------------------+-------------------------------+----------+----------+------------------+--------------------+

    For a better example of the content of the CSV file, please refer to the example CSV file provided in the NoisyCircuits repository within the noise_models directory.

    Args:
        calibration_data_file (str): The path to the CSV file containing the calibration data.
    
    Raises:
        TypeError: Raised when the input arguements are not of the expected type.
            - calibration_data_file should be a string representing the path to the CSV file.
            - basis_gates should be a list of lists of strings representing the basis gates for the quantum hardware.
        FileNotFoundError: If the specified CSV file is not found.
        ValueError: If the CSV file does not contain the required columns or if the data is not in the expected format.
    """
    def __init__(self, 
                 calibration_data_file:str,
                 basis_gates:list[list[str]])->None:
        """
        Constructor for the CreateNoiseModel class.
        """
        if not isinstance(calibration_data_file, str):
            raise TypeError("calibration_data_file must be a Path-like string")
        self.calibration_data_file = calibration_data_file
        try:
            self.calibration_data = pd.read_csv(calibration_data_file, delimiter=",")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{calibration_data_file}' not found. Please check the file path and try again.")
        if not isinstance(basis_gates, list) or not all(isinstance(gate_list, list) for gate_list in basis_gates) or not all(isinstance(gate, str) for gate_list in basis_gates for gate in gate_list):
            raise TypeError("basis_gates must be a list of lists of strings representing the basis gates for the quantum hardware")
        self.basis_gates = basis_gates

    def create_noise_model(self)->dict:
        """
        Method to create the noise model from the calibration data.

        Returns:
            dict: The noise model of the specified backend in dictionary format.
        """
        noise_model = NoiseModel()
        qubits = self.calibration_data["Qubit"].tolist()
        t1 = self.calibration_data["T1 (us)"].tolist()
        t2 = self.calibration_data["T2 (us)"].tolist()
        for q in range(len(qubits)):
            qubits[q] = int(qubits[q])
            t1[q] = float(t1[q]) * 1e-6
            t2[q] = float(t2[q]) * 1e-6
        for i, row in self.calibration_data.iterrows():
            p01 = float(row["Prob meas 0 prep 1"])
            p10 = float(row["Prob meas 1 prep 0"])
            single_qubit_gate_length = int(row["Single Qubit Gate Length (ns)"]) * 1e-9
            two_qubit_gate_length = row["Gate Length (ns)"].split(";")
            readout_matrix = np.array([[1 - p10, p10], [p01, 1 - p01]])
            noise_model.add_readout_error(readout_error(readout_matrix), [qubits[i]])
            for gate in self.basis_gates[0]:
                gate_error = int(row["{} Error".format(gate)])
                thermal_error = thermal_relaxation_error(t1[i], t2[i], single_qubit_gate_length)
                depolarization_error = depolarizing_error(gate_error, 1)
                noise_model.add_quantum_error(thermal_error.compose(depolarization_error), gate, [qubits[i]])
            for gate in self.basis_gates[1]:
                gate_error = row["{} Error".format(gate)]
                connected_qubits_errors = gate_error.split(";")
                for k in range(len(connected_qubits_errors)):
                    target, error = connected_qubits_errors[k].split(":")
                    target_qubit = int(target)
                    error = float(error)
                    gate_time = int(two_qubit_gate_length[k].split(":")[1]) * 1e-9
                    thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                        thermal_relaxation_error(t1[target_qubit], t2[target_qubit], gate_time)
                    )
                    depolarization_error = depolarizing_error(error, 2)
                    noise_model.add_quantum_error(thermal_error.compose(depolarization_error), gate, [qubits[i], target_qubit])
        return noise_model.to_dict(serializable=True)