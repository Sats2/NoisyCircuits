from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService


class GetNoiseModel:
    def __init__(self,
                 backend_name:str,
                 token:str,
                 channel:str="ibm_quantum_platform")->None:
        if not isinstance(backend_name, str):
            raise TypeError("backend_name must be a string")
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if not isinstance(channel, str):
            raise TypeError("channel must be a string")
        self.backend_name = backend_name
        self.token = token
        self.channel = channel

    def get_noise_model(self)->dict:
        service = QiskitRuntimeService(channel=self.channel, token=self.token)
        if not service.backends(self.backend_name):
            raise ValueError(f"Backend '{self.backend_name}' not found in user account. Please check the backend name or your account settings.")
        try:
            backend = service.backend(self.backend_name)
        except Exception as e:
            raise ValueError(f"Error retrieving backend '{self.backend_name}': {str(e)}")
        noise_data = NoiseModel.from_backend(backend)
        noise_model = noise_data.to_dict(serializable=True)
        return noise_model