import json
import pytest
from NoisyCircuits.utils.GetNoiseModel import GetNoiseModel
import os

api_key = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))["apikey"]

def test_token_type():
    """
    Test that a TypeError is raised when token is not a string.
    """
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez",
                      token=12345)
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez",
                      token=["token_string"])

def test_backend_name_type():
    """
    Test that a TypeError is raised when backend_name is not a string.
    """
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name=12345,
                      token=api_key)
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name=["ibm_fez"],
                      token=api_key)

def test_channel_type():
    """
    Test that a TypeError is raised when channel is not a string.
    """
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez",
                      token=api_key,
                      channel=12345)
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez",
                      token=api_key,
                      channel=["ibm_quantum_platform"])
        
def test_get_noise_model_invalid_backend():
    """
    Test that a ValueError is raised when an invalid backend_name is provided.
    """
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="invalid_backend_name",
                    token=api_key).get_noise_model()
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_perth",
                    token=api_key).get_noise_model()

def test_get_noise_model_invalid_token():
    """
    Test that a ValueError is raised when an invalid token is provided.
    """
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_fez",
                    token="invalid_token").get_noise_model()
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_fez",
                    token="").get_noise_model()

def test_get_noise_model_invalid_channel():
    """
    Test that a ValueError is raised when an invalid channel is provided.
    """
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_fez",
                    token=api_key,
                    channel="invalid_channel").get_noise_model()
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_fez",
                    token=api_key,
                    channel="").get_noise_model()