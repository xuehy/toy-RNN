#ifndef RNN_DEVICE_ALTERNATE_H_
#define RNN_DEVICE_ALTERNATE_H_

#define CUDA_KERNEL_LOOP(i, n) \
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i+= blockDim.x * gridDim.x)

const int RNN_CUDA_NUM_THREADS = 512;

inline int RNN_GET_BLOCKS(const int N)
{
  return (N + RNN_CUDA_NUM_THREADS - 1) / RNN_CUDA_NUM_THREADS;
}
#endif
