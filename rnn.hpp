#ifndef _RNN_H_
#define _RNN_H_

#include <cblas.h>
#include <memory>
#include <random>
#include <algorithm>
#include <utility>
#include "rnn_math.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "netS.pb.h"
#include "netD.pb.h"
#include <csignal>

using namespace std;

enum mode {CPU, GPU};

template <typename DTYPE>
class RNN
{
  int word_dim_;
  int hidden_dim_;
  int  bptt_truncate_;
  unique_ptr<DTYPE[]> U;
  unique_ptr<DTYPE[]> V;
  unique_ptr<DTYPE[]> W;

  DTYPE *dev_U, *dev_W, *dev_V;
  /**
   * cuda and cublas 
   */
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  void cublas_check();
  void cuda_check();
  /**
   * solver mode: CPU or GPU
   */

  mode solver_mode_;
  /**
   * some solver parameters
   */
  DTYPE lr_;
  int epoch_;
  /**
   * internal variable for output and state
   */
  unique_ptr<DTYPE[]> o_;
  unique_ptr<DTYPE[]> s_;

  /**
   * device memory pointers
   */
  DTYPE *dev_o_;
  DTYPE *dev_s_;
  DTYPE *dev_Vs_t;
  /**
   * gradient for V
   */
  unique_ptr<DTYPE[]> dV;
  DTYPE *dev_dV;
  /**
   * gradient for W
   */
  unique_ptr<DTYPE[]> dW;
  DTYPE *dev_dW;
  unique_ptr<DTYPE[]> dU;
  DTYPE *dev_dU;

  /**
   * some device memory pointers
   * that will be used
   * during forward_gpu 
   * and backward_gpu
   */
  
  /**
   * copy constructor is forbidden
   */
  RNN(const RNN& rnn);
  /**
   * assign constructor is forbidden
   */
  RNN& operator=(const RNN& rnn);
public:
  RNN(int word_dim, int hidden_dim, int bptt_trun);
  explicit RNN(string snapshot);
  ~RNN();
  /**
   * forward of RNN on CPU
   * Our RNN reads one-hot vectors as input
   * @x x should store a sentence where each word is a one-hot vector for simplicity we just use the non-zero index
  */
  void forward_cpu(vector <int> &x);
  void forward_gpu(vector <int> &x);
  /**
   * @x input vector
   * @return the vector of index of highest score
   */
  vector <int> predict(vector <int> &x);

  /**
   * @x traning samples
   * @y true labels
   * @return the total loss
   */
  DTYPE calculate_total_loss(vector <vector <int>> &x, vector <vector <int>> &y);

  /**
   * divide the total loss by the number of training samples
   */
  DTYPE calculate_loss(vector <vector <int>> &x, vector <vector <int>> &y);

  /**
   * backpropagation through time
   * @x a sentence
   * @y label
   */
  void bptt(vector <int> &x, vector <int> &y);
  void gradient_check(vector <int> &x, vector <int> &y, DTYPE h = 0.001, DTYPE err_thres = 0.01);

  /**
   * Perform one step of SGD
   */
  void sgd_step(vector <int> &x, vector <int> &y, DTYPE learning_rate);

  /**
   * @snapshot_interval store the net parameters in file every snapshot_interval epochs
   */
  void train(vector <vector <int>> &X_train, vector <vector <int>> &Y_train,
	     vector <vector <int>> &x_val, vector <vector <int>> &y_val,
	     DTYPE learning_rate = 0.005, int nepoch = 1000,
	     int evaluate_loss_after = 50, int val_after = 5000, int snapshot_interval = 50);

  /**
   * save the model parameters to file 'snapshot'
   */
  void write(string snapshot);
  /**
   * read parameters and create the model
   */
  void read(string snapshot);

  void set_mode(mode solver_mode);

  /**
   * initialize solver mode on GPU
   * if on CPU, do nothing
   * if on GPU, initalize device and allocate device memories
   */
  void initialize();

  /**
   * finalize solver mode on GPU
   * copy the device memory to host memory
   * such that the parameters can be saved
   */
  void finalize();
};



#include "rnn.cpp"
#endif

