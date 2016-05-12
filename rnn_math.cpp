#include "rnn_math.h"

void softmax(double *input, double *output, int dim)
{
  double maximum = *max_element(input, input + dim);
  copy(input, input + dim, output);
  for_each(output, output + dim, [&](double &x){x = x - maximum;});
  for_each(output, output + dim, [&](double &x){x = exp(x);});
  double sum = accumulate(output, output + dim, 0.0);
  for_each(output, output + dim, [&](double &x){x = x / sum;});
}
