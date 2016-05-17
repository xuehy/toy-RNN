#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename DTYPE>
void rnn_gpu_gemv(cublasHandle_t handle, CblasTranspose trans,
		  int m, int n,
		  const DTYPE *alpha,
		  const DTYPE *A, 
		  const DTYPE *x,
		  const DTYPE *beta,
		  DTYPE *y) {}

template <>
void rnn_gpu_gemv<double>(cublasHandle_t handle, CblasTranspose trans,
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
void rnn_gpu_gemv<float>(cublasHandle_t handle, CblasTranspose trans,
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

template <typename DTYPE>
void rnn_gpu_copy(cublasHandle_t handle, int N, DTYPE *X, DTYPE *Y) {}

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

template <typename DTYPE>
void rnn_gpu_scal<DTYPE>(cublasHandle_t handle, int N, const DTYPE *alpha, DTYPE *x) {}

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
