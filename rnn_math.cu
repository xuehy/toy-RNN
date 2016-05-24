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

template <>
void rnn_gpu_set<int>(cublasHandle_t handle, int N, const int *X, int *Y)
{
  cublasSetVector(N, sizeof(int), X, 1, Y, 1);
}

template <>
void rnn_gpu_set<double>(cublasHandle_t handle, int N, const double *X, double *Y)
{
  cublasSetVector(N, sizeof(double), X, 1, Y, 1);
}

template <>
void rnn_gpu_set<float>(cublasHandle_t handle, int N, const float *X, float *Y)
{
  cublasSetVector(N, sizeof(float), X, 1, Y, 1);
}

template <>
void rnn_gpu_get(cublasHandle_t handle, int N, const float *X, float *Y)
{
  cublasGetVector(N, sizeof(float), X, 1, Y, 1);
}

template <>
void rnn_gpu_get(cublasHandle_t handle, int N, const double *X, double *Y)
{
  cublasGetVector(N, sizeof(double), X, 1, Y, 1);
}


// cuda kernels
template <typename DTYPE>
__global__ void kernel_exp(DTYPE *input, DTYPE *output, int N)
{
  CUDA_KERNEL_LOOP(index,N)
    {
      output[index] = exp(input[index]);
    }
}

template <typename DTYPE>
__global__ void kernel_add_scalar(DTYPE alpha, DTYPE *input, DTYPE *output, int N)
{
  CUDA_KERNEL_LOOP(index, N)
    {
      output[index] = input[index] + alpha;
    }
}

template <typename DTYPE>
__global__ void kernel_sub_scalar(DTYPE *alpha, const DTYPE *input, DTYPE *output, const int N)
{
  CUDA_KERNEL_LOOP(index, N)
    {
      output[index] = input[index] - alpha[0];
    }
}

template <typename DTYPE>
__global__ void kernel_tanh(const DTYPE *input, DTYPE *output, int N)
{
  CUDA_KERNEL_LOOP(index, N)
    {
      output[index] = tanh(input[index]);
    }
}

template <typename DTYPE>
__global__ void kernel_div_scalar(DTYPE *alpha, DTYPE *input, DTYPE *output, const int N)
{
  CUDA_KERNEL_LOOP(index, N)
    {
      output[index] = input[index] / alpha[0];
    }
}

template <>
void rnn_gpu_tanh<float>(const int N, const float *X, float *Y)
{
  kernel_tanh<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(X, Y, N);
}

template <>
void rnn_gpu_tanh<double>(const int N, const double *X, double *Y)
{
  kernel_tanh<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(X, Y, N);
}

template <typename DTYPE>
__global__ void kernel_max(const DTYPE *input, DTYPE *out, const int N)
{
  CUDA_KERNEL_LOOP(index, 1)
    {
      DTYPE maxval = -FLT_MAX;
      for(int c = 0; c < N; ++c)
	maxval = max(input[c], maxval);
      *out = maxval;
    }
}

template <typename DTYPE>
__global__ void kernel_sum(const DTYPE *input, DTYPE *out, const int N)
{
  CUDA_KERNEL_LOOP(index, 1)
    {
      DTYPE sum = 0;
      for(int c = 0; c < N; ++c)
	sum += input[c];
      *out = sum;
    } 
}

// reduction sum needs detailed improvement
template <typename DTYPE>
__global__ void reduction_sum(const DTYPE *input, DTYPE *output, const int N)
{
      int tid = threadIdx.x;
      int bid = blockIdx.x;
      int index = bid * blockDim.x + tid;
      int block_start = bid * RNN_CUDA_NUM_THREADS;
      // shared memory is shared within the same block
      __shared__ DTYPE s_data[RNN_CUDA_NUM_THREADS];
      if(block_start + tid < N)
	s_data[tid] = input[index];
      else
	s_data[tid] = 0;
	
      __syncthreads();

      
      for(int i = RNN_CUDA_NUM_THREADS / 2; i > 0; i >>= 1)
	{
	  if (tid < i)
	    s_data[tid] = s_data[tid] + s_data[tid + i];
	  __syncthreads();
	}
      if (tid == 0)
	output[bid] = s_data[0];
}

/**
 * takes device pointers as parameters
 * @input 
 * @sum: the array of the same size as input
 */
template <>
void rnn_reduction_sum<float>(const float *input, float *sum, int N)
{
  reduction_sum<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(input, sum, N);
  N = RNN_GET_BLOCKS(N);
  if (N == 1) return;
  while ( N > 1 )
  {
    reduction_sum<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(sum, sum, N);
      N = RNN_GET_BLOCKS(N);
  }
}

template <>
void rnn_reduction_sum<double>(const double *input, double *sum, int N)
{
  reduction_sum<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(input, sum, N);
  N = RNN_GET_BLOCKS(N);
  if( 1 == N ) return;
  while ( N > 1 )
    {
      reduction_sum<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(sum, sum, N);
      N = RNN_GET_BLOCKS(N);
    }
}

// reduction sum needs detailed improvement
template <typename DTYPE>
__global__ void reduction_max(const DTYPE *input, DTYPE *output, const int N)
{
      int tid = threadIdx.x;
      int bid = blockIdx.x;
      int index = bid * blockDim.x + tid;
      int block_start = bid * RNN_CUDA_NUM_THREADS;
      // shared memory is shared within the same block
      __shared__ DTYPE s_data[RNN_CUDA_NUM_THREADS];
      if(block_start + tid < N)
	s_data[tid] = input[index];
      else
	s_data[tid] = -FLT_MAX;
	
      __syncthreads();

      
      for(int i = RNN_CUDA_NUM_THREADS / 2; i > 0; i >>= 1)
	{
	  if (tid < i)
	    s_data[tid] = max(s_data[tid], s_data[tid + i]);
	  __syncthreads();
	}
      if (tid == 0)
	output[bid] = s_data[0];
}

template <>
void rnn_reduction_max<float>(const float *input, float *output, int N)
{
  reduction_max<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(input, output, N);
  N = RNN_GET_BLOCKS(N);
  if (N == 1) return;
  while ( N > 1 )
  {
    reduction_max<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(output, output, N);
      N = RNN_GET_BLOCKS(N);
  }
}

template <>
void rnn_reduction_max<double>(const double *input, double *output, int N)
{
  reduction_max<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(input, output, N);
  N = RNN_GET_BLOCKS(N);
  if (N == 1) return;
  while ( N > 1 )
  {
    reduction_max<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(output, output, N);
      N = RNN_GET_BLOCKS(N);
  }
}

template <>
void rnn_gpu_softmax<float>(const int N, const float *X, float *Y, float *reduction)
{
  rnn_reduction_max<float>(X, reduction, N);
  kernel_sub_scalar<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(reduction, X, Y, N);
  kernel_exp<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(Y, Y, N);
  rnn_reduction_sum<float>(Y, reduction, N);
  kernel_div_scalar<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(reduction, Y, Y, N);

  // float *maxval;
  // cudaMalloc((void**)&maxval, sizeof(float));
  // kernel_max<float><<<1,1>>>(X, maxval, N);
  // kernel_sub_scalar<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(maxval, X, Y, N);

  // kernel_exp<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(Y, Y, N);
  // kernel_sum<float><<<1,1>>>(Y, maxval, N);

  // kernel_div_scalar<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(maxval, Y, Y, N);
  // cudaFree(maxval)
}

template <>
void rnn_gpu_softmax<double>(const int N, const double *X, double *Y, double *reduction)
{
  rnn_reduction_max<double>(X, reduction, N);
  kernel_sub_scalar<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(reduction, X, Y, N);
  kernel_exp<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(Y, Y, N);
  rnn_reduction_sum<double>(Y, reduction, N);
  kernel_div_scalar<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(reduction, Y, Y, N);
}

// == the following functions are layer-specific
// this function is ugly
template <typename DTYPE>
__global__ void kernel_softmax_grad(DTYPE *input, int *y, const int word_dim, const int T)
{
  CUDA_KERNEL_LOOP(index, T)
    {
      input[index*word_dim + y[index]] -= DTYPE(1.0);
    }
}

template <>
void softmax_grad_gpu<float>(float *input, int *y, const int word_dim, const int T)
{
  kernel_softmax_grad<float><<<RNN_GET_BLOCKS(T), RNN_CUDA_NUM_THREADS>>>(input, y, word_dim, T);
}

template <>
void softmax_grad_gpu<double>(double *input, int *y, const int word_dim, const int T)
{
  kernel_softmax_grad<double><<<RNN_GET_BLOCKS(T), RNN_CUDA_NUM_THREADS>>>(input, y, word_dim, T);
}

template <typename DTYPE>
__global__ void kernel_tanh_grad(DTYPE *input1, DTYPE *input2, DTYPE *output, int N)
{
  CUDA_KERNEL_LOOP(index, N)
    {
      output[index] = input1[index] * (1 - input2[index] * input2[index]);
    }
}

template <>
void tanh_grad_gpu<float>(float *input1, float *input2, float *output, int N)
{
  kernel_tanh_grad<float><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(input1, input2, output ,N);
}

template <>
void tanh_grad_gpu<double>(double *input1, double *input2, double *output, int N)
{
  kernel_tanh_grad<double><<<RNN_GET_BLOCKS(N), RNN_CUDA_NUM_THREADS>>>(input1, input2, output, N);
}

