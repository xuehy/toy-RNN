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

template <typename DTYPE>
void rnn_math_gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE Trans, int M, int N, DTYPE alpha, DTYPE *A, int lda,
	      DTYPE *X, DTYPE beta, DTYPE *Y){}

template <>
void rnn_math_gemv<double>(CBLAS_ORDER order, CBLAS_TRANSPOSE Trans, int M, int N, double alpha,
		      double *A, int lda, double *X, double beta, double *Y)
{
  cblas_dgemv(order, Trans, M, N, alpha, A, lda, X, 1, beta, Y, 1);
}

template <>
void rnn_math_gemv<float>(CBLAS_ORDER order, CBLAS_TRANSPOSE Trans, int M, int N, float alpha,
		      float *A, int lda, float *X, float beta, float *Y)
{
  cblas_sgemv(order, Trans, M, N, alpha, A, lda, X, 1, beta, Y, 1);
}


template <typename DTYPE>
void rnn_math_copy(int N, DTYPE *X, DTYPE *Y){}

template <>
void rnn_math_copy<double>(int N, double *X, double *Y)
{
  cblas_dcopy(N, X, 1, Y, 1);
}

template <>
void rnn_math_copy<float>(int N, float *X, float *Y)
{
  cblas_scopy(N, X, 1, Y, 1);
}


template <typename DTYPE>
void rnn_math_scal(int N, DTYPE alpha, DTYPE *X){exit(1);}

template <>
void rnn_math_scal<double>(int N, double alpha, double *X)
{
  cblas_dscal(N, alpha, X, 1);
}

template <>
void rnn_math_scal<float>(int N, float alpha, float *X)
{
  cblas_sscal(N, alpha, X, 1);
}


template <typename DTYPE>
void rnn_math_ger(CBLAS_ORDER order,
		  int M, int N, DTYPE alpha, DTYPE *X, DTYPE *Y, DTYPE *A, int lda){}

template <>
void rnn_math_ger<double>(CBLAS_ORDER order, 
		  int M, int N, double alpha, double *X, double *Y, double *A, int lda)
{
  cblas_dger(order, M, N, alpha, X, 1, Y, 1, A, lda);
}

template <>
void rnn_math_ger<float>(CBLAS_ORDER order, 
		  int M, int N, float alpha, float *X, float *Y, float *A, int lda)
{
  cblas_sger(order, M, N, alpha, X, 1, Y, 1, A, lda);
}

template <typename DTYPE>
void rnn_math_axpy(int N, DTYPE alpha, DTYPE *X, DTYPE *Y) {}

template <>
void rnn_math_axpy<double>(int N, double alpha, double *X, double *Y)
{
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template <>
void rnn_math_axpy<float>(int N, float alpha, float *X, float *Y)
{
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

#endif
