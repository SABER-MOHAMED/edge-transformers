#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>
#include <cstddef>

// dot product Function
double dot_product(const std::vector < double > & a,
  const std::vector < double > & b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

// matrix-vector multiplication
std::vector<double> matrix_vector_multiply(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<double> &vector
) {
    std::vector<double> result(matrix.size(), 0.0);

    for(size_t i = 0; i < matrix.size(); i++) {
        for(size_t j = 0; j < matrix[i].size(); j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

// softMax function, takes a vector and returns the SoftMax probabilities
std::vector < double > softmax(const std::vector < double > & input) {
  // declare output vector
  std::vector < double > output(input.size());

  // compute max value for numerical stability, mainly to avoid overflow
  double max_val = * std::max_element(input.begin(), input.end());
  double sum = 0.0;

  for (size_t i = 0; i < input.size(); i++) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }

  for (size_t i = 0; i < output.size(); ++i) {
    output[i] /= sum;
  }
  return output;
}


#endif