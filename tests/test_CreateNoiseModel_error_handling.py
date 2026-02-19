import os
import pathlib
import pandas as pd
import pytest
from NoisyCircuits.utils.CreateNoiseModel import CreateNoiseModel


def test_calibration_data_file_type():
    """
    Test that the calibration_data_file parameter raises TypeError for invalid types.
    """
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=12345)
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=["list_instead_of_string"])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file={"dict_instead_of_string": "value"})

def test_calibration_data_file_not_found():
    """
    Test that a FileNotFoundError is raised when the specified CSV file is not found.
    """
    with pytest.raises(FileNotFoundError):
        CreateNoiseModel(calibration_data_file="non_existent_file.csv")
    with pytest.raises(FileNotFoundError):
        CreateNoiseModel(calibration_data_file="noise_models/non_existent_file.csv")

def test_basis_gates_type():
    """
    Test that the basis_gates parameter raises TypeError for invalid types.
    """
    file_path = os.path.join(pathlib.Path(__file__).parent.parent, "noise_models/sample_calibration_data.csv")
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates="invalid_basis_gates")
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=12345)
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=[["x", "sx"], "cz"])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=[["x", "sx"], [12345]])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=[[12345], ["cz"]])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=[["x", "sx"], ["cz"], "extra_element"])
    with pytest.raises(TypeError):
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=[["x", "sx"], ["cz"], ["extra_element"]])

def test_valid_calibration_data():
    """
    Test that a ValueError is raised when the CSV file does not contain the required columns or if the data is not in the expected format.
    """
    file_path = os.path.join(pathlib.Path(__file__).parent.parent, "noise_models/junk_data.csv")
    basis_gates = [["x", "sx"], ["cz"]]
    with pytest.raises(ValueError):
        data = pd.DataFrame(columns=["Something", "Else"], data=[["junk", "data"], ["junk1", "data2"]])
        data.to_csv(file_path, index=False)
        CreateNoiseModel(calibration_data_file=file_path, basis_gates=basis_gates).create_noise_model()
        os.remove(file_path)