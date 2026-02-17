from pyavaframe.outputs import QuantityOfInterest, Outputs
import numpy as np
from pyavaframe.utils import FrictionModel
from pyavaframe.utils import SimulationFlags
from pyavaframe.simulator import Simulator, UncertainParameters
from pyavaframe.utils import runSimulation
import fnmatch
import os
import time
import pandas as pd
import pickle
import rasterio

input_file_path = r"Run_Case_Wog/"

# friction = FrictionModel(model_name="Voellmy",
#                friction_parameters={
#                    "mu": [True, [0.02, 0.5]],
#                    "xi": [True, [400, 4000]]
#                })
# # friction = FrictionModel(
# #     model_name = "samosAT",
# #     friction_parameters={
# #         "mu" : [True, [0.02, 0.5]],
# #         "tau": [False, 140],
# #         "RS0": [True, [0.01, 0.5]],
# #         "kappa" : [True, [0.35, 0.75]],
# #         "RS" : [True, [0.05, 0.15]],
# #         "BS" : [True, [3.5, 4.5]]
# #     }
# # )

# simulator = Simulator(
#     input_directory=input_file_path,
#     num_simulations=500,
#     friction_model=friction,
#     uncertainity_parameters=[UncertainParameters.FRICTION_PARAMETERS],
#     limits=[],
#     simulation_flags=SimulationFlags(res_type=["ppr", "FV", "pft", "FT"], 
#                                      release_thickness_from_file=True,
#                                      mesh_cell_size=5),
#     serial_simulation=False,
#     num_nodes=1,
#     num_cpu_cores=50,
#     use_all_cores=False,
#     rng_seed=24
# )

# # Specify file storage locations here!
# data_directory = "/Users/adam-ukj7r05xnu2fywx/result_data_files/" # Location to store model design and response.
# output_directory = "/Users/adam-ukj7r05xnu2fywx/data_files_voellmy/" # Location to store all simulation results from Avaframe Simulations.
# simulator.generate_files(output_directory=output_directory)
# with open("Simulator_Voellmy.pkl", "wb") as f:
#     pickle.dump(simulator, f)
# simulator.save_uncertainity_parameters(output_directory=data_directory, file_name="Voellmy_Design")
# simulator.run_simulations()

# output = Outputs(simulator, [QuantityOfInterest.flowV, QuantityOfInterest.flowT, QuantityOfInterest.peak_hMax])
# aggregate_data = output.get_aggregated_outputs(threshold=0.1, unit="km", save_output=True, file_name="Voellmy_Response", output_directory=data_directory)


def calculate_ia(input_directory, threshold=0.1):
    prefix = "*_pft*"
    peak_file_directory = input_directory + "Outputs/com1DFA/peakFiles/"
    peak_file = [f for f in os.listdir(peak_file_directory) if fnmatch.fnmatch(f, prefix)]
    for file in peak_file:
        read_file = peak_file_directory + file
        with rasterio.open(read_file) as src:
            valid_cells = np.where(src.read(1) >= threshold, 1, 0)
        if "ent" in read_file:
            ia_ent = np.sum(valid_cells) * (src.res[0] ** 2)
        elif "null" in read_file:
            ia_null = np.sum(valid_cells) * (src.res[0] ** 2)
    return (ia_ent, ia_null)

file_list = ["/Users/adam-ukj7r05xnu2fywx/data_voellmy_true/", "/Users/adam-ukj7r05xnu2fywx/data_voellmy_true/local_com1DFACfg.ini"]

runSimulation(file_list)
data = calculate_ia(file_list[0])
print("Entrained Impact Area: ", data[0])
print("Null Impact Area: ", data[1])

data_dict = {
    "params": [0.23, 1000],
    "ia_entrained": data[0],
    "ia_null": data[1]
}
with open("result_data_files/true_data_voellmy.pkl", "wb") as f:
    pickle.dump(data_dict, f)