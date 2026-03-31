#include <complex>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using complex128 = std::complex<double>;


int get_gpu_device_count(){
    return omp_get_num_devices();
}

static inline void apply_H_gate(complex128* __restrict__ state, std::size_t q, std::size_t num_qubits, const unsigned short thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    // const std::size_t step = stride << 1;
    constexpr double inv_sqrt_2 = 0.70710678118655;

    #pragma omp target teams distribute parallel for thread_limit(thread_count) 
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;

        const complex128 s0 = state[i];
        const complex128 s1 = state[j];

        state[i] = inv_sqrt_2 * (s0 + s1);
        state[j] = inv_sqrt_2 * (s0 - s1);
    }
}

static inline void apply_RX_gate(complex128* __restrict__ state, std::size_t q, std::size_t num_qubits, double theta, const unsigned short thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    // const std::size_t step = stride << 1;

    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);

    #pragma omp target teams distribute parallel for thread_limit(thread_count) 
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;

        const complex128 s0 = state[i];
        const complex128 s1 = state[j];

        state[i] = complex128(cosine * s0.real() + sine * s1.imag(), cosine * s0.imag() - sine * s1.real());
        state[j] = complex128(sine * s0.imag() + cosine * s1.real(), cosine * s1.imag() - sine * s0.real());
    }
}

static inline void apply_CZ_gate(complex128* const __restrict__ state, std::size_t q1, std::size_t q2, std::size_t num_qubits, const unsigned short thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t iters = dim >> 2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t target_mask = (1ULL << q1) | (1ULL << q2);
    
    #pragma omp target teams distribute parallel for thread_limit(thread_count)
    for (std::size_t i = 0; i < iters; ++i){
        std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        state[pos | target_mask] = -state[pos | target_mask];
    }
}

void run_circuit_kernel(complex128* __restrict__ host_state, const double* __restrict__ host_angles, std::size_t num_angles, std::size_t num_qubits, unsigned int depth, const unsigned short thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;

    #pragma omp target data map(to: host_angles[0:num_angles]) map(tofrom: host_state[0:dim])
    {
        for (std::size_t q = 0; q < num_qubits; q++){
            apply_H_gate(host_state, q, num_qubits, thread_count);
        }
        for (unsigned int d = 0; d < depth; d++){   
            for (std::size_t q = 0; q < num_qubits; q++){
                apply_RX_gate(host_state, q, num_qubits, host_angles[d * num_qubits + q], thread_count);
            }
            for (std::size_t q = 0; q < num_qubits - 1; q++){
                apply_CZ_gate(host_state, q, q + 1, num_qubits, thread_count);
            }
        }
    }
}

void run_circuit_gpu(py::array_t<double> input_angles, py::size_t num_qubits, py::array_t<complex128> output_state, const unsigned short thread_count){
    if (get_gpu_device_count() == 0){
        throw std::runtime_error("No GPU devices available for simulation.");
    }
    py::buffer_info angles_buf = input_angles.request();
    double* angles_ptr = static_cast<double*>(angles_buf.ptr);
    const std::size_t num_angles = angles_buf.size;
    const unsigned int depth = num_angles / num_qubits;

    py::buffer_info state_buf = output_state.request();
    auto* state = static_cast<complex128*>(state_buf.ptr);
    state[0] = complex128(1.0, 0.0);

    run_circuit_kernel(state, angles_ptr, num_angles, num_qubits, depth, thread_count);
}

PYBIND11_MODULE(run_gpu, m){
    m.doc() = "C++ Module to simulate circuits with GPU acceleration";
    m.def("run_circuit_gpu", &run_circuit_gpu, "Run a Quantum Circuit on a GPU");
    m.def("get_gpu_device_count", &get_gpu_device_count, "Get the number of GPU devices available");
}