// multiple head attention mechanism
#include <iostream>
#include <vector>
#include <iomanip>
#include "math_utils.h"


// core attention function
std::vector < double > attention(const std::vector < double > & query,
  const std::vector < std::vector < double > > & keys,
    const std::vector < std::vector < double > > & values) {

  // calculate attention scores
  std::vector < double > scores;

  for (size_t i = 0; i < keys.size(); ++i) {
    scores.push_back(dot_product(query, keys[i]));
  }
  // apply softmax to scores
  std::vector < double > weights = softmax(scores);

  // compute weighted sum of values using weights
  std::vector < double > output(values[0].size(), 0.0); // initialize output vector with zeros
  for (size_t i = 0; i < values.size(); ++i) {
    for (size_t j = 0; j < values[i].size(); ++j) {
      output[j] += weights[i] * values[i][j];
    }
  }
  return output;
}


// multipple single-heads approach
std::vector <double> multiple_attention(
    const std::vector<double> &query,
    const std::vector<std::vector<double>> &keys,
    const std::vector<std::vector<double>> &values,
    const std::vector<std::vector<double>> &W_Q,
    const std::vector<std::vector<double>> &W_K,
    const std::vector<std::vector<double>> &W_V
) {
    // project query, keys, values
    std::vector<double> projected_query = matrix_vector_multiply(W_Q, query);
    // reserve memory upfront in order to improve performance
    std::vector<std::vector<double>> projected_keys, projected_values;
    projected_keys.reserve(keys.size());
    projected_values.reserve(values.size());

    // combined loop to project keys and values
    const size_t num_sequences = std::max(keys.size(), values.size());
    for (size_t i = 0; i < num_sequences; i++) {
        if (i < keys.size()) {
            // project keys
            projected_keys.emplace_back(matrix_vector_multiply(W_K, keys[i]));
        }
        if (i < values.size()) {
            // project values
            projected_values.emplace_back(matrix_vector_multiply(W_V, values[i]));
        }
    }

    // compute attention using the projected vectors
    return attention(projected_query, projected_keys, projected_values);
}

// main function for testing
int main() {
    // example query, keys, values
    std::vector<double> query = {1.0, 0.0, 1.0};
    std::vector<std::vector<double>> keys = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 1.0}
    };
    std::vector<std::vector<double>> values = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };

    // example projection
    std::vector<std::vector<double>> W_Q = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    std::vector<std::vector<double>> W_K = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    std::vector<std::vector<double>> W_V = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    std::vector<double> output = multiple_attention(query, keys, values, W_Q, W_K, W_V);
    std::cout << "Multi-head Attention Output: ";
    for (const auto &val : output) {
        std::cout << std::setprecision(4) << val << " ";
    }
    std::cout << std::endl;

    return 0;
}