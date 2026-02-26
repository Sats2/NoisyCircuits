"""
This module allows the user to create the raw noise model dictionary from user provided calibration data (CSV file) from an actual Quantum Hardware without the requirement for any tokens. This raw noise model can be provided to the QuantumCircuit or BuildQubitGateModel modules for post-processing. The BuildQubitGateModel module returns the post-processed noise operators for each standardized gate on each qubit whereas the QuantumCircuit module uses the post-processed noise model to perform quantum circuit simulations under noise.

Additionally, the user must specify the basis gates for the quantum hardware in order to create the noise model. The basis gates must be provided in a list of lists format where the first list contains the single qubit basis gates and the second list contains the two qubit basis gates. For example, if the single qubit basis gates are "x", "sx" and "rz" and the two qubit basis gate is "cz", then the basis gates should be provided as [["x", "sx", "rz"], ["cz"]].

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

Example:
    >>> from NoisyCircuits.utils.CreateNoiseModel import CreateNoiseModel
    >>> calibration_file_path = "path/to/calibration_data.csv"
    >>> basis_gates = [["x", "sx", "rz"], ["cz"]]
    >>> noise_model = CreateNoiseModel(calibration_data_file=calibration_file_path, basis_gates=basis_gates).create_noise_model()
"""

from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError
import pandas as pd
import numpy as np


class CreateNoiseModel:
    """
    A class to create a noise model from user provided calibration data. The calibration data should be provided in a CSV file format with specific columns for qubit properties and gate errors. The class reads the calibration data, processes it, and constructs a noise model that can be used for quantum circuit simulations.

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
        relaxation_time_anomaly_index = np.where(np.array(t2) > 2 * np.array(t1))[0]
        if len(relaxation_time_anomaly_index) > 0:
            for index in relaxation_time_anomaly_index:
                print(r"Warning: Found relaxation time anomaly for qubit {} with $T_2 \geq 2T_1$. Setting $T_2 = 2T_1$.".format(qubits[index]))
                t2[index] = 2 * t1[index]
        mean_errors = {}
        for gate in self.basis_gates[0]:
            mean_errors[gate] = self.calibration_data["{} Error".format(gate)].mean()
        for q in range(len(qubits)):
            qubits[q] = int(qubits[q])
            t1[q] = t1[q] * 1e-6
            t2[q] = t2[q] * 1e-6
        for i, row in self.calibration_data.iterrows():
            p01 = float(row["Prob meas 0 prep 1"])
            p10 = float(row["Prob meas 1 prep 0"])
            single_qubit_gate_length = int(row["Single Qubit Gate Length (ns)"]) * 1e-9
            two_qubit_gate_length = row["Gate Length (ns)"].split(";")
            readout_matrix = np.array([[1 - p10, p10], [p01, 1 - p01]])
            noise_model.add_readout_error(ReadoutError(readout_matrix), [qubits[i]])
            for gate in self.basis_gates[0]:
                gate_error = row["{} Error".format(gate)]
                if np.isnan(gate_error):
                    gate_error = mean_errors[gate]
                    print("Warning: Using Mean Error Rate {:.6f} for gate {} on qubit {} due to missing values.".format(gate_error, gate, qubits[i]))
                thermal_error = thermal_relaxation_error(t1[i], t2[i], single_qubit_gate_length)
                depol_error = depolarizing_error(float(gate_error), 1)
                noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i]])

            for gate in self.basis_gates[1]:
                gate_error = row["{} Error".format(gate)]
                try:
                    connected_qubits_errors = gate_error.split(";")
                except:
                    connected_qubits_errors = []
                if len(connected_qubits_errors) == len(two_qubit_gate_length):
                    for j in range(len(connected_qubits_errors)):
                        target_qubits, target_error = connected_qubits_errors[j].split(":")
                        gate_target, gate_length = two_qubit_gate_length[j].split(":")
                        if int(target_qubits) != int(gate_target):
                            print("Found Mismazch in target qubits for gate {} for qubits {} and {}".format(gate, target_qubits, gate_target))
                        target_qubits = int(target_qubits)
                        gate_time = int(gate_length) * 1e-9
                        gate_error = float(target_error)
                        thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                            thermal_relaxation_error(t1[target_qubits], t2[target_qubits], gate_time)
                        )
                        depol_error = depolarizing_error(gate_error, 2)
                        noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[target_qubits]])
                elif len(connected_qubits_errors) != 0:
                    applied_qubits = []
                    error_mean = 0.0
                    qubit_connections = [idx.split(":")[0] for idx in two_qubit_gate_length]
                    for j in range(len(connected_qubits_errors)):
                        target_qubits, target_error = connected_qubits_errors[j].split(":")
                        gate_target, gate_time = two_qubit_gate_length[j].split(":")
                        if int(gate_target) != int(target_qubits):
                            for k in range(len(qubit_connections)):
                                gate_target, gate_time = two_qubit_gate_length[k].split(":")
                                if int(gate_target) == int(target_qubits):
                                    break
                        target_qubits = int(target_qubits)
                        gate_time = int(gate_time) * 1e-9
                        gate_error = float(target_error)
                        error_mean += gate_error
                        applied_qubits.append(target_qubits)
                        thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                            thermal_relaxation_error(t1[target_qubits], t2[target_qubits], gate_time)
                        )
                        depol_error = depolarizing_error(gate_error, 2)
                        noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[target_qubits]])
                    error_mean = error_mean / len(connected_qubits_errors)
                    for idx in qubit_connections:
                        idx = int(idx)
                        if idx not in applied_qubits:
                            print("Warning: No error data found for gate {} on qubit {} connected to qubit {}. Applying mean error rate of {:.6f}".format(gate, qubits[i], qubits[idx], error_mean))
                            target_qubits = idx
                            gate_time = two_qubit_gate_length[qubit_connections.index(str(idx))].split(":")[1]
                            gate_time = int(gate_time) * 1e-9
                            thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                                thermal_relaxation_error(t1[target_qubits], t2[target_qubits], gate_time)
                            )
                            depol_error = depolarizing_error(error_mean, 2)
                            noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[target_qubits]])
                else:
                    print("No Error Data found for gate {} on qubit {}".format(gate, qubits[i]))
                    applied_qubits = []
                    error_mean = 0.0
                    if len(self.basis_gates[1]) > 1:
                        print("Attempting to apply error data from other two qubit gates for qubit {}".format(qubits[i]))
                        for g in self.basis_gates[1]:
                            if g == gate:
                                continue
                            gate_error = row["{} Error".format(g)]
                            try:
                                connected_qubits_errors = gate_error.split(";")
                            except:
                                connected_qubits_errors = []
                            if len(connected_qubits_errors) == len(two_qubit_gate_length) or len(connected_qubits_errors) != 0:
                                break
                        if len(connected_qubits_errors) != 0:
                            qubit_connections = [idx.split(":")[0] for idx in two_qubit_gate_length]
                            for j in range(len(connected_qubits_errors)):
                                target_qubits, target_error = connected_qubits_errors[j].split(":")
                                for k in range(len(two_qubit_gate_length)):
                                    gate_target, gate_time = two_qubit_gate_length[k].split(":")
                                    if int(gate_target) == int(target_qubits):
                                        break
                                target_qubits = int(target_qubits)
                                gate_time = int(gate_time) * 1e-9
                                gate_error = float(target_error)
                                error_mean += gate_error
                                applied_qubits.append(target_qubits)
                                thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                                    thermal_relaxation_error(t1[target_qubits], t2[target_qubits], gate_time)
                                )
                                depol_error = depolarizing_error(gate_error, 2)
                                noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[target_qubits]])
                            error_mean = error_mean / len(connected_qubits_errors)
                            for idx in qubit_connections:
                                if int(idx) not in applied_qubits:
                                    print(f"Warning: No error data found for gate {gate} on qubit {qubits[i]} connected to qubit {qubits[int(idx)]}. Applying mean error rate of {error_mean:.6f}")
                                    target_qubits = int(idx)
                                    gate_time = two_qubit_gate_length[qubit_connections.index(idx)].split(":")[1]
                                    gate_time = int(gate_time) * 1e-9
                                    thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                                        thermal_relaxation_error(t1[int(idx)], t2[int(idx)], gate_time)
                                    )
                                    depol_error = depolarizing_error(error_mean, 2)
                                    noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[int(idx)]])
                        else:
                            print("No error data found for any two qubit gates on qubit {}. Applying maximum error rate for gate {}".format(qubits[i], gate))
                            for j in range(len(two_qubit_gate_length)):
                                target_qubits, gate_time = two_qubit_gate_length[j].split(":")
                                target_qubits = int(target_qubits)
                                gate_time = int(gate_time) * 1e-9
                                thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                                    thermal_relaxation_error(t1[target_qubits], t2[target_qubits], gate_time)
                                )
                                depol_error = depolarizing_error(1.0, 2)
                                noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[target_qubits]])
                    else:
                        print("Only a single two qubit gate found in basis gates. Applying maximum error rate for gate {} on qubit {}".format(gate, qubits[i]))
                        for j in range(len(two_qubit_gate_length)):
                            target_qubits, gate_time = two_qubit_gate_length[j].split(":")
                            target_qubits = int(target_qubits)
                            gate_time = int(gate_time) * 1e-9
                            thermal_error = thermal_relaxation_error(t1[i], t2[i], gate_time).tensor(
                                thermal_relaxation_error(t1[target_qubits], t2[target_qubits], gate_time)
                            )
                            depol_error = depolarizing_error(1.0, 2)
                            noise_model.add_quantum_error(thermal_error.compose(depol_error), gate, [qubits[i], qubits[target_qubits]])
        return noise_model.to_dict(serializable=True)

#TODO: Add functionality to read-in backend information from quantum hardware.
import requests

class GetNoiseModel:
    """
    This class allows the user to obtain the noise model from the calibration data of IBM Quantum Hardware using IBM Rest API. This class retreives the calibration data of the specified backend and processes it to create the noise model in dictionary format. 

    A valid IBM Quantum API token and CRN is required to access the calibration data of the backend.

    Args:
        backend_name (str): The name of the IBM Quantum backend.
        token (str): The IBM Quantum API token.
        service_crn (str): The CRN of the IBM Quantum service instance.

    Raises:
        TypeError: If backend_name, token or service_crn is not a string.
        ValueError: If there is an issue connecting to the IBM Quantum API or if the backend is not found in the user's account.
    """
    def __init__(self,
                 backend_name:str,
                 token:str,
                 service_crn:str)->None:
        """
        Constructor for the GetNoiseModel class.
        """
        if not isinstance(backend_name, str):
            raise TypeError("backend_name must be a string")
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if not isinstance(service_crn, str):
            raise TypeError("service_crn must be a string")
        self.backend_name = backend_name
        self.token = token
        self.service_crn = service_crn
        self._api_url = "https://quantum.cloud.ibm.com/api/v1/backends/{}/properties".format(backend_name)
        self._get_IAM_token()
        self._headers = {
            "Accept": "application/json",
            "IBM-API-Version": "2026-02-15",
            "Authorization": "Bearer {}".format(self._access_token),
            "Service-CRN": self.service_crn
        }

    def _get_IAM_token(self)->None:
        """
        Private method to obtain the IAM token for authentication with the IBM Quantum API using the provided token.
        """
        iam_response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data = "grant_type=urn:ibm:params:oauth:grant-tpye:apikey&apikey={}".format(self.token),
        )
        self._access_token = iam_response.json()["access_token"]

    def _get_calibration_json(self)->None:
        """
        Private method of the class to obtain the raw calibration data in JSON format from the IBM Quantum API for the specified backend.

        Raises:
            ValueError: If the API request fails or if the backend is not found in the user's account, an error is raised with the detailed error message from the API response.
        """
        response = requests.request(
            "GET", 
            self._api_url,
            headers = self._headers,
        )
        if response.status_code != 200:
            raise ValueError("Failed to retrieve calibration data from IBM Quantum API. Detailed error: {}".format(response.text))
        self.calibration_json = response.json()

    def _convert_json_to_csv(self, 
                             save_csv:bool=False,
                             destination:str=None,
                             file_name:str=None
                             )->None:
        """
        Private method of the class that converts the raw calibration data in the JSON format obtained from IBM Quantum API to a dataframe and optionally saves it as a CSV file.

        Args:
            save_csv (bool, optional): Flag to decide whether to save the converted calibration data as a CSV file. Defaults to False.
            desitnation (str, optional): The directory where the CSV file should be saved if required by the user. If None and the csv needs to be saved, the file will be saved in the current working directory. Defaults to None.
            
        """
        data = pd.DataFrame()