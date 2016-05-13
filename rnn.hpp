#ifndef _RNN_H_
#define _RNN_H_

#include <cblas.h>
#include <memory>
#include <random>
#include <algorithm>
#include <utility>
#include "rnn_math.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>

#include <csignal>
#include "net.pb.h"
using namespace std;
class RNN
{
  int word_dim_;
  int hidden_dim_;
  double  bptt_truncate_;
  unique_ptr<double[]> U;
  unique_ptr<double[]> V;
  unique_ptr<double[]> W;
  /**
   * some solver parameters
   */
  double lr_;
  int epoch_;
  /**
   * internal variable for output and state
   */
  unique_ptr<double[]> o_;
  unique_ptr<double[]> s_;
  /**
   * gradient for V
   */
  unique_ptr<double[]> dV;
  /**
   * gradient for W
   */
  unique_ptr<double[]> dW;
  unique_ptr<double[]> dU;
  /**
   * copy constructor is forbidden
   */
  RNN(const RNN& rnn);
  /**
   * assign constructor is forbidden
   */
  RNN& operator=(const RNN& rnn);
public:
  RNN(int word_dim, int hidden_dim, double bptt_trun);
  RNN(string snapshot);
  /**
   * forward of RNN on CPU
   * Our RNN reads one-hot vectors as input
   * @x x should store a sentence where each word is a one-hot vector for simplicity we just use the non-zero index
  */
  void forward_cpu(vector <int> &x);
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
  double calculate_total_loss(vector <vector <int>> &x, vector <vector <int>> &y);

  /**
   * divide the total loss by the number of training samples
   */
  double calculate_loss(vector <vector <int>> &x, vector <vector <int>> &y);

  /**
   * backpropagation through time
   * @x a sentence
   * @y label
   */
  void bptt(vector <int> &x, vector <int> &y);
  void gradient_check(vector <int> &x, vector <int> &y, double h = 0.001, double err_thres = 0.01);

  /**
   * Perform one step of SGD
   */
  void sgd_step(vector <int> &x, vector <int> &y, double learning_rate);

  /**
   * @snapshot_interval store the net parameters in file every snapshot_interval epochs
   */
  void train(vector <vector <int>> &X_train, vector <vector <int>> &Y_train,
	     vector <vector <int>> &x_val, vector <vector <int>> &y_val,
	     double learning_rate = 0.005, int nepoch = 1000,
	     int evaluate_loss_after = 50, int val_after = 5000, int snapshot_interval = 50);

  /**
   * save the model parameters to file 'snapshot'
   */
  void write(string snapshot);
  /**
   * read parameters and create the model
   */
  void read(string snapshot);


};


#endif
