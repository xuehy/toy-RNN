#include "rnn_math.hpp"

template <>
void rnn_gpu_gemv<double>(cublasHandle_t handle, CBLAS_TRANSPOSE trans,
		  int m, int n,
		  const double *alpha,
		  const double *A, 
		  const double *x,
		  const double *beta,
		  double *y)
{
  cublasOperation_t cublasTrans = (trans == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasDgemv(handle, cublasTrans, n, m, alpha, A, n, x, 1, beta, y, 1);
}

template <>
void rnn_gpu_gemv<float>(cublasHandle_t handle, CBLAS_TRANSPOSE trans,
		  int m, int n,
		  const float *alpha,
		  const float *A, 
		  const float *x,
		  const float *beta,
		  float *y)
{
  cublasOperation_t cublasTrans = (trans == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemv(handle, cublasTrans, n, m, alpha, A, n, x, 1, beta, y, 1);
}

template <>
void rnn_gpu_copy<double>(cublasHandle_t handle, int N, double *X, double *Y)
{
  cublasDcopy(handle, N, X, 1, Y, 1);
}

template <>
void rnn_gpu_copy<float>(cublasHandle_t handle, int N, float *X, float *Y)
{
  cublasScopy(handle, N, X, 1, Y, 1);
}

template <>
void rnn_gpu_scal<double>(cublasHandle_t handle, int N, const double *alpha, double *x)
{
  cublasDscal(handle, N, alpha, x, 1);
}

template <>
void rnn_gpu_scal<float>(cublasHandle_t handle, int N, const float *alpha, float *x)
{
  cublasSscal(handle, N, alpha, x, 1);
}

template <>
void rnn_gpu_ger<double>(cublasHandle_t handle, 
			 int M, int N, double *alpha, double *X, double *Y, double *A)
{
  cublasDger(handle, N, M, alpha, Y, 1, X, 1, A, N);
}

template <>
void rnn_gpu_ger<float>(cublasHandle_t handle, 
			 int M, int N, float *alpha, float *X, float *Y, float *A)
{
  cublasSger(handle, N, M, alpha, Y, 1, X, 1, A, N);
}

template <>
void rnn_gpu_axpy<double>(cublasHandle_t handle, int N, double *alpha, double *X, double *Y)
{
  cublasDaxpy(handle, N, alpha, X, 1, Y, 1);
}

template <>
void rnn_gpu_axpy<float>(cublasHandle_t handle, int N, float *alpha, float *X, float *Y)
{
  cublasSaxpy(handle, N, alpha, X, 1, Y, 1);
}