#include "TypeDefs.hpp"

using real_matrix = std::vector<std::vector<double>>;

std::vector<real_matrix> get_measurement_error_matrices(const py::dict& noise_dictionary){
    std::vector<real_matrix> noise_matrix_list;
    for (auto item : noise_dictionary){
        int qubit = py::cast<int>(item.first);
        auto noise_matrix = py::cast<py::array>(item.second);
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(noise_matrix);
        auto a = arr.unchecked<2>();
        real_matrix noise_matrix_converted(a.shape(0), std::vector<double>(a.shape(1)));
        for (unsigned short i = 0; i < a.shape(0); i++){
            #pragma GCC unroll 2
            for (unsigned short j = 0; j < a.shape(1); j++){
                noise_matrix_converted[i][j] = a(i, j);
            }
        }
        noise_matrix_list.push_back(std::move(noise_matrix_converted));
    }
    return noise_matrix_list;
}

static inline void apply_measurement_error_for_qubit(double * __restrict__ probabilities, real_matrix measurement_noise_operator, const int qubit, const std::size_t num_qubits, const uint8 num_threads){
    double m00 = measurement_noise_operator[0][0];
    double m01 = measurement_noise_operator[0][1];
    double m10 = measurement_noise_operator[1][0];
    double m11 = measurement_noise_operator[1][1];
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << qubit;
    #pragma omp parallel for num_threads(num_threads)
    for (std::size_t p = 0; p < (dim >> 1); ++p){
        const std::size_t i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const double p0 = probabilities[i];
        const double p1 = probabilities[j];
        probabilities[i] = m00 * p0 + m01 * p1;
        probabilities[j] = m10 * p0 + m11 * p1;
    }
}

void apply_measurement_error(py::array_t<double> probabilities_array, py::dict measurement_noise, py::list qubit_list, std::size_t num_qubits, uint8 num_threads){
    std::vector<int> qubit_list_cpp;
    for (auto item : qubit_list){
        qubit_list_cpp.push_back(py::cast<int>(item));
    }
    std::vector<real_matrix> measurement_error_matrices  = get_measurement_error_matrices(measurement_noise);
    py::buffer_info probs_info = probabilities_array.request();
    auto * probabilities = static_cast<double*>(probs_info.ptr);
    for (std::size_t q = 0; q < qubit_list_cpp.size(); q++){
        int qubit_index = qubit_list_cpp[q];
        real_matrix measurement_noise_operator = measurement_error_matrices[q];
        apply_measurement_error_for_qubit(probabilities, measurement_noise_operator, qubit_index, num_qubits, num_threads);
    }
}

PYBIND11_MODULE(measurement_error_applicator, m){
    m.doc() = "Module that applies measurement error to a probability distribution over quantum states given a noise model.";
    m.def("apply_measurement_error", &apply_measurement_error, "Applies measurement error to a probability distribution over quantum states", py::arg("probabilities_array"), py::arg("measurement_noise"), py::arg("qubit_list"), py::arg("num_qubits"), py::arg("num_threads"));
}