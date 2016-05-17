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

template <typename DTYPE>
void softmax(DTYPE *input, DTYPE *output, int dim)
{
  DTYPE maximum = *max_element(input, input + dim);
  copy(input, input + dim, output);
  for_each(output, output + dim, [&](DTYPE &x){x = x - maximum;});
  for_each(output, output + dim, [&](DTYPE &x){x = exp(x);});
  DTYPE sum = accumulate(output, output + dim, 0.0);
  for_each(output, output + dim, [&](DTYPE &x){x = x / sum;});
}


#endif
