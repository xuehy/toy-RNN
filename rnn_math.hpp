#ifndef _RNN_MATH_H_
#define _RNN_MATH_H_
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cfloat>
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_alternate.hpp"
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
		   DTYPE *X, DTYPE beta, DTYPE *Y);

template <typename DTYPE>
void rnn_math_copy(int N, DTYPE *X, DTYPE *Y);

template <typename DTYPE>
void rnn_math_scal(int N, DTYPE alpha, DTYPE *X);

template <typename DTYPE>
void rnn_math_ger(CBLAS_ORDER order,
		  int M, int N, DTYPE alpha, DTYPE *X, DTYPE *Y, DTYPE *A, int lda);

template <typename DTYPE>
void rnn_math_axpy(int N, DTYPE alpha, DTYPE *X, DTYPE *Y)
;
template <typename DTYPE>
void rnn_gpu_gemv(cublasHandle_t handle, CBLAS_TRANSPOSE trans,
		  int m, int n,
		  const DTYPE *alpha,
		  const DTYPE *A, 
		  const DTYPE *x,
		  const DTYPE *beta,
		  DTYPE *y);

template <typename DTYPE>
void rnn_gpu_copy(cublasHandle_t handle, int N, DTYPE *X, DTYPE *Y);

template <typename DTYPE>
void rnn_gpu_scal(cublasHandle_t handle, int N, const DTYPE *alpha, DTYPE *x);

template <typename DTYPE>
void rnn_gpu_ger(cublasHandle_t handle, 
		 int M, int N, DTYPE *alpha, DTYPE *X, DTYPE *Y, DTYPE *A);

template <typename DTYPE>
void rnn_gpu_axpy(cublasHandle_t handle, int N, DTYPE *alpha, DTYPE *X, DTYPE *Y);

template <typename DTYPE>
void rnn_gpu_set(cublasHandle_t handle, int N, const DTYPE *X, DTYPE *Y);

template <typename DTYPE>
void rnn_gpu_get(cublasHandle_t handle, int N, const DTYPE *X, DTYPE *Y);

template <typename DTYPE>
void rnn_gpu_tanh(const int N, const DTYPE *X, DTYPE *Y);

template <typename DTYPE>
void rnn_gpu_softmax(const int N, const DTYPE *X, DTYPE *Y);
#include "rnn_math.hpp"
#endif
