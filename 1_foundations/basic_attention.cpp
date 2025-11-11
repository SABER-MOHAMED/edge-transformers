#include <iostream>
#include <vector>
#include <iomanip> // For std::setprecision
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

// main function
int main() {
  // Query, Value, and Key matrices are fundamental components of the attention mechanism in transformer models.
  // test data
  std::vector < double > query = {
    1.0,
    0.0,
    1.0
  };
  std::vector < std::vector < double >> keys = {
    {
      1.0,
      0.0,
      0.0
    },
    {
      0.0,
      1.0,
      0.0
    },
    {
      1.0,
      0.0,
      1.0
    }
  };
  std::vector < std::vector < double >> values = {
    {
      1.0,
      0.0
    },
    {
      10.0,
      0.0
    },
    {
      100.0,
      5.0
    }
  };

  // compute attention output
  std::vector < double > output = attention(query, keys, values);
  // print output
  std::cout << "Basic attention Output: ";
  for (const auto & val: output) {
    std::cout << std::setprecision(4) << val << " ";
  }
  std::cout << std::endl;

  return 0;
}