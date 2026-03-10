import os
import pathlib
import pandas as pd
import pytest
import json
from NoisyCircuits.utils.CreateNoiseModel import CreateNoiseModel, GetNoiseModel


"""
Unit Tests for error handling in the CreateNoiseModel class within the module.
"""
def test_invalid_calibration_file_path_type():
    """
    Test whether the CreateNoiseModel class raises a TypeError when the calibration_data_file is not a string.
    """
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=12345, 
                         basis_gates=[["x"], ["cz"]])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=["path/to/calibration_data.csv"], 
                         basis_gates=[["x"], ["cz"]])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file={"path": "to/calibration_data.csv"}, 
                         basis_gates=[["x"], ["cz"]])

def test_invalid_basis_gates_type():
    """
    Test whether the CreateNoiseModel class raises a TypeError when the basis_gates is not a list of a list of strings.
    """
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file="path/to/calibration_data.csv", 
                         basis_gates="x, cz")
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file="path/to/calibration_data.csv",
                         basis_gates=["x", "cz"])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file="path/to/calibration_data.csv",
                         basis_gates={"single_qubit": ["x"], "two_qubit":["cx"]})
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file="path/to/calibration_data.csv",
                         basis_gates=[["x"], "cz"])
    with pytest.raises(TypeError):
        CreateNoiseModel(caibration_data_file="path/to/calibration_data.csv",
                         basis_gates=123456)
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file="path/to/calibration_data.csv",
                         basis_gates=[[12345], [1254]])

def test_calibration_file_not_found_error():
    """
    Test whether the CreateNoiseModel class raises a FileNotFoundError when the calibration_data_file is non-existant.
    """
    with pytest.raises(FileNotFoundError):
        CreateNoiseModel(calibration_data_file="non_existent_file.csv", 
                         basis_gates=[["x"], ["cz"]])

def test_invalid_gate_in_backend_data():
    """
    Test whether the CreateNoiseModel class raises a ValueError when the wrong basis gates are provided.
    """
    file_path = os.path.join(pathlib.Path(__file__).parent, "../noise_models/Sample_Noise_Model_Heron_QPU.csv")
    with pytest.raises(ValueError):
        CreateNoiseModel(calibration_data_file=file_path,
                         basis_gates=[["x", "sx", "rz"], ["cz", "cx"]])
    with pytest.raises(ValueError):
        CreateNoiseModel(calibration_data_file=file_path,
                         basis_gates=[["x", "sx", "rz"], ["cz", "cy"]])

def test_missing_column_names():
    """
    Test whether the CreateNoiseModel class raises a ValueError when the calibration_data_file does not conform to the prescribed CSV format.
    """
    file_path = os.path.join(pathlib.Path(__file__).parent, "../noise_models/Sample_Noise_Model_Heron_QPU.csv")
    data = pd.read_csv(file_path)
    data.drop(columns=["T1 (us)"], inplace=True)
    modified_file_path = os.path.join(pathlib.Path(__file__).parent, "../noise_models/Modified_Sample_data.csv")
    data.to_csv(modified_file_path, index=False)
    with pytest.raises(ValueError):
        CreateNoiseModel(calibration_data_file=modified_file_path,
                         basis_gates=[["x", "sx", "rz"], ["cz"]])
    os.remove(modified_file_path)


"""
Unit Tests for error handling in the GetNoiseModel class within the module.
"""
def test_invalid_backend_type():
    """
    Test whether the GetNoiseModel class raises a TypeError when the backend_name is not a string.
    """
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name=12345, token="valid", service_crn="valid")
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name=["ibm_perth"], token="valid", service_crn="valid")
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name={"name": "ibm_perth"}, token="valid", service_crn="valid")

def test_invalid_token_type():
    """
    Test whether the GetNoiseModel class raises a TypeError when the token is not a string.
    """
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez", token=12345, service_crn="valid")
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez", token=["valid_token"], service_crn="valid")
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez", token={"token": "valid_token"}, service_crn="valid")

def test_invalid_service_crn_type():
    """
    Test whether the GetNoiseModel class raises a TypeError when the service_crn is not a string.
    """
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez", token="valid_token", service_crn=12345)
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez", token="valid_token", service_crn=["valid_service_crn"])
    with pytest.raises(TypeError):
        GetNoiseModel(backend_name="ibm_fez", token="valid_token", service_crn={"crn": "valid_service_crn"})

def test_fail_api_retrieval():
    """
    Test whether the GetNoiseModel class raises a ValueError when the API retrieval fails due to invalid token or service CRN.
    """
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_fez", token="invalid_token", service_crn="valid_service_crn").get_noise_model()
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_fez", token="valid_token", service_crn="invalid_service_crn").get_noise_model()

@pytest.mark.localonly
def test_backend_not_found():
    """
    Test whether the GetNoiseModel class raises a ValueError when the backend_name points to a backend that is not available.
    """
    api_json = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))
    token = api_json["apikey"]
    service_crn = api_json["service-crn"]
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="non_existent_backend", token=token, service_crn=service_crn).get_noise_model()
    with pytest.raises(ValueError):
        GetNoiseModel(backend_name="ibm_miami", token=token, service_crn=service_crn).get_noise_model()

@pytest.mark.localonly
def test_save_csv_invalid_data_type():
    """
    Test whether the GetNoiseModel class raises a TypeError when the save_csv flag is not a boolean.
    """
    api_json = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))
    token = api_json["apikey"]
    service_crn = api_json["service-crn"]
    modeller = GetNoiseModel(backend_name="ibm_fez", token=token, service_crn=service_crn)
    with pytest.raises(TypeError):
        modeller.get_noise_model(save_csv="something", destination="somewhere", file_name="something")
    with pytest.raises(TypeError):
        modeller.get_noise_model(save_csv=12, destination="somewhere", file_name="something")

@pytest.mark.localonly
def test_save_csv_invalid_destination_type():
    """
    Test whether the GetNoiseModel class raises a TypeError when the destination is not a string.
    """
    api_json = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))
    token = api_json["apikey"]
    service_crn = api_json["service-crn"]
    modeller = GetNoiseModel(backend_name="ibm_fez", token=token, service_crn=service_crn)
    with pytest.raises(TypeError):
        modeller.get_noise_model(save_csv=True, destination=12345,
        file_name="something")
    with pytest.raises(TypeError):
        modeller.get_noise_model(save_csv=True, destination=["somewhere"],
        file_name="something")

@pytest.mark.localonly
def test_save_csv_invalid_destination_location():
    """
    Test whether the GetNoiseModel class raises a NotADirectoryError when the destination directory is not found.
    """
    api_json = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))
    token = api_json["apikey"]
    service_crn = api_json["service-crn"]
    modeller = GetNoiseModel(backend_name="ibm_fez", token=token, service_crn=service_crn)
    with pytest.raises(NotADirectoryError):
        modeller.get_noise_model(save_csv=True, destination="/non_existent_directory", file_name="something.csv")

@pytest.mark.localonly
def test_save_csv_invalid_file_name_type():
    """
    Test whether the GetNoiseModel class raises a TypeError when the file_name is not a string.
    """
    api_json = json.load(open(os.path.join(os.path.expanduser("~"), "ibm_api.json"), "r"))
    token = api_json["apikey"]
    service_crn = api_json["service-crn"]
    modeller = GetNoiseModel(backend_name="ibm_fez", token=token, service_crn=service_crn)
    with pytest.raises(TypeError):
        modeller.get_noise_model(save_csv=True, destination=None, file_name=12345)
    with pytest.raises(TypeError):
        modeller.get_noise_model(save_csv=True, destination=None, file_name=["something.csv"])