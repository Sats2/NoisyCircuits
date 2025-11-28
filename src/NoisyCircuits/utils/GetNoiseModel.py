"""
This module allows the user to get the raw noise model dictionary from the calibration data of IBM quantum hardware (user's choice) using their IBM token. This raw noise model can be provided to the QuantumCircuit or BuildQubitGateModel modules for post-processing. The BuildQubitGateModel module returns the post-processed noise operators for each standardized gate on each qubit whereas the QuantumCircuit module uses the post-processed noise model to perform quantum circuit simulations under noise.

Example:
    >>> from NoisyCircuits.utils.GetNoiseModel import GetNoiseModel
    >>> noise_model = GetNoiseModel(backend_name=backend_name, token=token).get_noise_model()
"""
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService


class GetNoiseModel:
    """
    A class to retrieve the noise model of a specified IBM Quantum backend.
    """
    def __init__(self,
                 backend_name:str,
                 token:str,
                 channel:str="ibm_quantum_platform")->None:
        """
        Constructor for GetNoiseModel class.
        
        Args:
            backend_name (str): The name of the IBM Quantum backend.
            token (str): The IBM Quantum API token.
            channel (str): The IBM Quantum channel to use. Default is "ibm_quantum_platform".

        Raises:
            TypeError: If backend_name, token or channel is not a string.
            ValueError: If channel is not one of the accepted values.
        """
        if not isinstance(backend_name, str):
            raise TypeError("backend_name must be a string")
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if not isinstance(channel, str):
            raise TypeError("channel must be a string")
        if channel not in ["ibm_quantum_platform", "ibm_cloud"]:
            raise ValueError("channel must be either 'ibm_quantum_platform' or 'ibm_cloud'")
        self.backend_name = backend_name
        self.token = token
        self.channel = channel

    def get_noise_model(self)->dict:
        """
        Method to retrieve the noise model from the specified backend.

        Raises:
            ValueError: If the backend_name is invalid or not found in the user's account,
                        or if there is an issue connecting to the IBM Quantum Runtime Service.
        
        Returns:
            dict: The noise model of the specified backend in dictionary format.
        """
        try:
            service = QiskitRuntimeService(channel=self.channel, token=self.token)
        except Exception as e:
            raise ValueError(f"Error connecting to IBM Quantum Runtime Service: {str(e)}")
        if not service.backends(self.backend_name):
            raise ValueError(f"Backend '{self.backend_name}' not found in user account. Please check the backend name or your account settings.")
        try:
            backend = service.backend(self.backend_name)
        except Exception as e:
            raise ValueError(f"Error retrieving backend '{self.backend_name}': {str(e)}")
        noise_data = NoiseModel.from_backend(backend)
        noise_model = noise_data.to_dict(serializable=True)
        return noise_model