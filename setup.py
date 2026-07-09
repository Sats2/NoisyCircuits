"""Build script for the compiled pybind11 extension modules.

All package metadata lives in pyproject.toml; this file only defines the
C++ extensions, which cannot be expressed declaratively.
"""

from setuptools import setup, Extension
import pybind11
import os
import shlex

# fsanitize=address is used for debugging;
system_gomp = "/lib/x86_64-linux-gnu/libgomp.so.1"
cpp_flags = ["-O2", "-march=native", "-mtune=native", "-funroll-loops", "-fcf-protection=none", "-fno-stack-protector"]
omp_flags = ["-fopenmp"]
target_flags = ['-foffload=nvptx-none']
# GCC OpenMP offload can crash with some LTO configurations; keep LTO disabled by default.
lto_flags = shlex.split(os.environ.get("NOISYCIRCUITS_LTO_FLAGS", "-fno-lto"))

ext_modules = [
    Extension(
        "simulator",
        ["./src/NoisyCircuits/utils/custom/src/Simulator.cpp"],
        include_dirs = [pybind11.get_include()],
        language = "c++",
        extra_compile_args = cpp_flags + omp_flags,
        extra_link_args = omp_flags
        ),
    Extension(
        "simulator_mpi",
        ["./src/NoisyCircuits/utils/custom/src/SimulatorMPI.cpp"],
        include_dirs = [pybind11.get_include()],
        extra_compile_args = cpp_flags + omp_flags,
        extra_link_args = omp_flags
    ),
    Extension(
        "measurement_error_applicator",
        ["./src/NoisyCircuits/utils/custom/src/MeasurementErrorApplicator.cpp"],
        include_dirs = [pybind11.get_include()],
        language = "c++",
        extra_compile_args = cpp_flags + omp_flags,
        extra_link_args = omp_flags
    ),
]

setup(ext_modules=ext_modules)
