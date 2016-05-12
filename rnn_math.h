#ifndef _RNN_MATH_H_
#define _RNN_MATH_H_
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace std;
/**
 * softmax function
 * @input input vector
 * @output output vector
 * @dim dimension of the vector
 */
void softmax(double *input, double *output, int dim);

#endif
