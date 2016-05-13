#include "rnn.hpp"

RNN::RNN(int word_dim, int hidden_dim, double bptt_truncate)
{
  word_dim_ = word_dim;
  hidden_dim_ = hidden_dim;
  bptt_truncate_ = bptt_truncate;
  epoch_ = 0;
  lr_ = 0.005;
  // == initialize model parameters ==
  
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<double> dis1(-1.0/sqrt(word_dim_), 1.0/sqrt(word_dim_));
  uniform_real_distribution<double> dis2(-1.0/sqrt(hidden_dim_), 1.0/sqrt(hidden_dim_));
  unique_ptr<double[]> U_temp(new double [hidden_dim_ * word_dim_]);
  unique_ptr<double[]> V_temp(new double [hidden_dim_ * word_dim_]);
  unique_ptr<double[]> W_temp(new double [hidden_dim_ * hidden_dim_]);

  unique_ptr<double[]> dU_temp(new double [hidden_dim_ * word_dim_]);
  unique_ptr<double[]> dV_temp(new double [hidden_dim_ * word_dim_]);
  unique_ptr<double[]> dW_temp(new double [hidden_dim_ * hidden_dim_]);

  U = move(U_temp);
  V = move(V_temp);
  W = move(W_temp);

  dU = move(dU_temp);
  dV = move(dV_temp);
  dW = move(dW_temp);

  fill(dU.get(), dU.get() + hidden_dim_ * word_dim_, 0.0);
  fill(dV.get(), dV.get() + hidden_dim_ * word_dim_, 0.0);
  fill(dW.get(), dW.get() + hidden_dim_ * hidden_dim_, 0.0);
  
  for_each(U.get(), U.get() + hidden_dim_ * word_dim_, [&](double &x){x = dis1(gen);});
  for_each(V.get(), V.get() + hidden_dim_ * word_dim_, [&](double &x){x = dis2(gen);});
  for_each(W.get(), W.get() + hidden_dim_ * hidden_dim_, [&](double &x){x = dis2(gen);});

}

RNN::RNN(string snapshot)
{
  read(snapshot);
}
/**
 * @x every element in x represents a word, the value indicates the non-0 index of the one-hot vector
 */
void RNN::forward_cpu(vector <int> &x)
{
  int T = x.size(); // T: length of the sentence
   
  // @s state vector
  // @o output vector 
  unique_ptr<double[]> s(new double[(T + 1) * hidden_dim_]);
  unique_ptr<double[]> o(new double[T * word_dim_]);

  fill(s.get(), s.get() + (T + 1) * hidden_dim_, 0.0);
  fill(o.get(), o.get() + T * word_dim_, 0.0);

  unique_ptr<double[]> Vs_t(new double[word_dim_]);
  double *s_t_1, *s_t, *U_t, *o_t;
  
  // s[t] = tanh( U^T * x[t] + w * s[t-1] )
  // when t = 0
  int t = 0;
  fill(Vs_t.get(), Vs_t.get() + word_dim_, 0.0);

  // add the x[0]'th row of U with W * s[-1]
  // s_t_1 : pointer to the start of s[-1] = s[T]
  s_t_1 = s.get() + (T) * hidden_dim_;
  //
  s_t = s.get() + t * hidden_dim_;
  U_t = U.get() + x[t] * hidden_dim_;
  o_t = o.get() + t * word_dim_;
  // copy U^T*x[t] to s[t]
  copy(U_t, U_t + hidden_dim_, s_t);
  // s[t] = s[t] + W * s[t-1]
  // notice
  cblas_dgemv(CblasRowMajor, CblasNoTrans, hidden_dim_, hidden_dim_, 1.0, W.get(), hidden_dim_, s_t_1, 1.0, 1.0, s_t, 1.0);
  // s[t] = tanh( s[t] )
  for_each(s_t, s_t + hidden_dim_, [&](double &x){x = tanh(x);});
  // Vs_t = V * s[t]
  // notice
  cblas_dgemv(CblasRowMajor, CblasNoTrans, word_dim_, hidden_dim_, 1.0, V.get(), hidden_dim_, s_t, 1.0, 0.0, Vs_t.get(), 1.0);
  // softmax: o[t] = softmax(Vs_t)
  softmax(Vs_t.get(), o_t, word_dim_);

  // for each time step t >= 1
  for(t = 1; t < T; ++t)
    {
      // empty the V * s[t] vector
      fill(Vs_t.get(), Vs_t.get() + word_dim_, 0.0);
      // add the x[t]'th row of U with W * s[t-1]
      // s_t_1 : pointer to the start of s[t-1]
      s_t_1 = s.get() + (t-1) * hidden_dim_;
      //
      s_t = s.get() + t * hidden_dim_;
      U_t = U.get() + x[t] * hidden_dim_;
      o_t = o.get() + t * word_dim_;
      // copy U^T*x[t] to s[t]
      copy(U_t, U_t + hidden_dim_, s_t);
      // s[t] = s[t] + W * s[t-1]
      cblas_dgemv(CblasRowMajor, CblasNoTrans, hidden_dim_, hidden_dim_, 1.0, W.get(), hidden_dim_, s_t_1, 1.0, 1.0, s_t, 1.0);
      // s[t] = tanh( s[t] )
      for_each(s_t, s_t + hidden_dim_, [&](double &x){x = tanh(x);});
      // Vs_t = V * s[t]
      cblas_dgemv(CblasRowMajor, CblasNoTrans, word_dim_, hidden_dim_, 1.0, V.get(), hidden_dim_, s_t,1.0, 0.0, Vs_t.get(), 1.0);
      softmax(Vs_t.get(), o_t, word_dim_);
    }

  s_ = move(s);
  o_ = move(o);
  
}

vector <int> RNN::predict(vector <int> &x)
{
  vector <int> result;
  forward_cpu(x);
  for(size_t i = 0; i < x.size(); ++i)
    {
      double *o_i = o_.get() + i * word_dim_;
      auto iter = max_element(o_i,o_i + word_dim_);
      int idx = static_cast<int>(iter - o_i);
      result.push_back(idx);
    }
  return result; 
}

/**
 * total loss of the entire sequence
 */
double RNN::calculate_total_loss(vector <vector <int>> &x, vector <vector <int>> &y)
{
  double L = 0.0;
  // for ith training sample
  for(size_t i = 0; i < y.size(); ++i)
    {
      forward_cpu(x[i]);
      // for jth word
      double loss = 0.0;
      for(size_t j = 0; j < y[i].size(); ++j)
	loss += log(o_[j * word_dim_ + y[i][j]]);
      L += -1 * loss;
    }
  return L;
}

double RNN::calculate_loss(vector <vector <int>> &x, vector <vector <int>> &y)
{
  int N = 0;
  for(size_t i = 0; i < y.size(); ++i)
    N += y[i].size();
  return calculate_total_loss(x,y) / N;
}

void RNN::bptt(vector <int> &x, vector <int> &y)
{
  // clear dU dV dW
  fill(dU.get(), dU.get() + hidden_dim_ * word_dim_, 0.0);
  fill(dV.get(), dV.get() + hidden_dim_ * word_dim_, 0.0);
  fill(dW.get(), dW.get() + hidden_dim_ * hidden_dim_, 0.0);

  int T = y.size();
  // perform forward propagation
  forward_cpu(x);

  unique_ptr<double[]> delta_o_ptr(new double[T * word_dim_]);
  copy(o_.get(), o_.get() + T * word_dim_, delta_o_ptr.get());

  double *delta_o = delta_o_ptr.get();
  for(int i = 0; i < T; ++i)
    delta_o[i * word_dim_ + y[i]] -= 1;


  unique_ptr<double[]> ds(new double[hidden_dim_]);
  fill(ds.get(), ds.get() + hidden_dim_, 0.0);
  
  // for every time step (word)
  for(int t = T - 1; t >= 0; t--)
    {
      // dV = (o_t - y_t) x s_t
      // gradient for V (correct)
      cblas_dger(CblasRowMajor, word_dim_, hidden_dim_, 1.0, delta_o + t * word_dim_, 1.0, s_.get() + t * hidden_dim_, 1.0, dV.get(), hidden_dim_);

      // // gradient for W
      double *delta_o_t = delta_o + t * word_dim_;
      fill(ds.get(), ds.get() + hidden_dim_, 0.0);
      // V^T * delta_o_t -> ds

      // CblasTrans: hidden_dim_
      cblas_dgemv(CblasRowMajor, CblasTrans, word_dim_, hidden_dim_, 1.0, V.get(), hidden_dim_, delta_o_t, 1.0, 0.0, ds.get(), 1.0);
      double *s_t = s_.get() + t * hidden_dim_;
      // ds * ( 1 - s_t ** 2) -> ds
      transform(ds.get(), ds.get() + hidden_dim_, s_t, ds.get(), [](double &x, double &y){return x * (1 - y * y);});
      for(int bptt_step = t; bptt_step >= max(0, t - (int)bptt_truncate_); bptt_step --)
      	{
	  double *s_bptt_step_1 = (bptt_step - 1 < 0)? s_.get() + T * hidden_dim_ : s_.get() + (bptt_step - 1) * hidden_dim_;
      	  // dW
      	  cblas_dger(CblasRowMajor, hidden_dim_, hidden_dim_, 1.0, ds.get(), 1.0, s_bptt_step_1, 1.0, dW.get(), hidden_dim_);
      	  // gradient for U
	  
      	  double *dU_step = dU.get() + x[bptt_step] * hidden_dim_;
      	  transform(dU_step, dU_step + hidden_dim_, ds.get(), dU_step, [](double &x, double &y){return x + y;});
      	  // update ds
	  unique_ptr<double[]> ws(new double[hidden_dim_]);
      	  cblas_dgemv(CblasRowMajor, CblasTrans, hidden_dim_, hidden_dim_, 1.0, W.get(), hidden_dim_, ds.get(), 1.0, 0.0, ws.get(), 1.0);
      	  transform(ws.get(), ws.get() + hidden_dim_, s_bptt_step_1, ds.get(), [&](double &a, double &b){return a * (1 - b * b);});
      	}
      
    }
}

void RNN::gradient_check(vector <int> &x, vector <int> &y, double h, double err_thres)
{
  bptt(x,y);
  cout << "Performing gradient check for parameter V with size " << hidden_dim_ * word_dim_ << endl;

  vector <vector <int>> X;
  vector <vector <int>> Y;
  X.push_back(x);
  Y.push_back(y);

  // check gradients for V
  for(int ix = 0; ix < hidden_dim_ * word_dim_; ++ix)
    {
      double original_value = V[ix];
      V[ix] = original_value + h;
      double grad_plus = calculate_total_loss(X, Y);
      V[ix] = original_value - h;
      double grad_minus = calculate_total_loss(X, Y);
      double estimated_gradient = (grad_plus - grad_minus) / (2 * h);
      V[ix] = original_value;
      double backprop_gradient = dV[ix];
      double relative_error = abs(backprop_gradient - estimated_gradient) / (abs(backprop_gradient) + abs(estimated_gradient));
      if (relative_error > err_thres)
	{
	  cout << "Gradient Check ERROR: parameter=V ix=" << ix << endl;
	  cout << "+h Loss: " << grad_plus << endl;
	  cout << "-h Loss: " << grad_minus << endl;
	  cout << "Estimated_gradient: " << estimated_gradient << endl;
	  cout << "Backpropagation gradient: " << backprop_gradient << endl;
	  cout << "Relative Error: " << relative_error << endl;
	  return ;
	}
    }
  cout << "Gradient check for parameter V passed" << endl;

  // check gradients for W
  for(int ix = 0; ix < hidden_dim_ * hidden_dim_; ++ix)
    {
      double original_value = W[ix];
      W[ix] = original_value + h;
      double grad_plus = calculate_total_loss(X, Y);
      W[ix] = original_value - h;
      double grad_minus = calculate_total_loss(X, Y);
      double estimated_gradient = (grad_plus - grad_minus) / (2 * h);
      W[ix] = original_value;
      double backprop_gradient = dW[ix];
      double relative_error = abs(backprop_gradient - estimated_gradient) / (abs(backprop_gradient) + abs(estimated_gradient));
      if (relative_error > err_thres)
	{
	  cout << "Gradient Check ERROR: parameter=W ix=" << ix << endl;
	  cout << "+h Loss: " << grad_plus << endl;
	  cout << "-h Loss: " << grad_minus << endl;
	  cout << "Estimated_gradient: " << estimated_gradient << endl;
	  cout << "Backpropagation gradient: " << backprop_gradient << endl;
	  cout << "Relative Error: " << relative_error << endl;
	  return;
	}
    }
  cout << "Gradient check for parameter W passed" << endl;

  // check gradients for U
  for(int ix = 0; ix < word_dim_ * hidden_dim_; ++ix)
    {
      double original_value = U[ix];
      U[ix] = original_value + h;
      double grad_plus = calculate_total_loss(X, Y);
      U[ix] = original_value - h;
      double grad_minus = calculate_total_loss(X, Y);
      double estimated_gradient = (grad_plus - grad_minus) / (2 * h);
      U[ix] = original_value;
      double backprop_gradient = dU[ix];
      double relative_error = abs(backprop_gradient - estimated_gradient) / (abs(backprop_gradient) + abs(estimated_gradient));
      if (relative_error > err_thres)
	{
	  cout << "Gradient Check ERROR: parameter=U ix=" << ix << endl;
	  cout << "+h Loss: " << grad_plus << endl;
	  cout << "-h Loss: " << grad_minus << endl;
	  cout << "Estimated_gradient: " << estimated_gradient << endl;
	  cout << "Backpropagation gradient: " << backprop_gradient << endl;
	  cout << "Relative Error: " << relative_error << endl;
	  return ;
	}
    }
  cout << "Gradient check for parameter U passed" << endl;
}

void RNN::sgd_step(vector <int> &x, vector <int> &y, double learning_rate)
{
  bptt(x,y);
  cblas_daxpy(word_dim_ * hidden_dim_, -learning_rate, dU.get(), 1, U.get(), 1);
  cblas_daxpy(word_dim_ * hidden_dim_, -learning_rate, dV.get(), 1, V.get(), 1);
  cblas_daxpy(hidden_dim_ * hidden_dim_, -learning_rate, dW.get(), 1, W.get(), 1);
}

void RNN::train(vector <vector <int>> &X_train, vector <vector <int>> &Y_train,
		vector <vector <int>> &x_val, vector <vector <int>> &y_val,
		double learning_rate, int nepoch, int evaluate_loss_after, int val_after,
		int snapshot_interval)
{
  // for every epoch
  time_t rawtime;
  struct tm *timeinfo;
  char buf[80];

  cout << "start training..." << endl;
  double loss_last = 10000.0;

  // this means the model is not at initial state
  // it is loaded from snapshots
  if (epoch_ != 0) {
    cout << "continuing training from epoch: " << epoch_ << endl;
    cout << "learning rate set to " << lr_ << endl;
    learning_rate = lr_;
    double val_loss = calculate_loss(x_val, y_val);
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", timeinfo);
    cout << buf << "validation loss = " << val_loss << endl;
  }
  else
    lr_ = learning_rate;
  cout << "continuing..." << endl;
  for(int epoch = epoch_; epoch < nepoch; epoch ++)
    {
      if (epoch % snapshot_interval == 0)
	{
	  string snapshot_filename = "rnnmodel_" + to_string(epoch) + ".snapshot";
	  write(snapshot_filename);
	}
      if (epoch % evaluate_loss_after == 0)
	{
	  double loss = calculate_loss(X_train, Y_train);
	  time(&rawtime);
	  timeinfo = localtime(&rawtime);
	  strftime(buf, sizeof(buf), "%Y-%m-%d %X", timeinfo);
	  cout << buf << "  Loss after" << " epoch=" << epoch << ": " << loss << endl;
	  if (loss > loss_last)
	    {
	      learning_rate *= 0.5;
	      lr_ = learning_rate;
	      cout << "   [notice] set learning rate to " << learning_rate << endl;
	    }
	  loss_last = loss;
	}
      if (epoch % val_after == 0)
        {
	  double val_loss = calculate_loss(x_val, y_val);
	  time(&rawtime);
	  timeinfo = localtime(&rawtime);
	  strftime(buf, sizeof(buf), "%Y-%m-%d %X", timeinfo);
	  cout << buf << " epoch=" << epoch << " validation loss = " << val_loss << endl;
	}

      for(size_t i = 0; i < Y_train.size(); ++i)
	{
	  sgd_step(X_train[i], Y_train[i], learning_rate);
	  
	}
      epoch_ = epoch + 1;
    }
  write("model_trained.snapshot");
}

void RNN::write(string snapshot)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  NetParamter net;
  net.set_hidden_dim(hidden_dim_);
  net.set_word_dim(word_dim_);
  net.set_learingrate(lr_);
  net.set_bptt_truncate(bptt_truncate_);
  net.set_epoch(epoch_);
  for(int i = 0; i < word_dim_ * hidden_dim_; ++i)
    {
      net.add_u(U[i]);
      net.add_v(V[i]);
    }
  for(int i = 0; i < hidden_dim_ * hidden_dim_; ++i)
    {
      net.add_w(W[i]);
    }

  fstream output(snapshot, ios::binary | ios::out | ios::trunc);
  net.SerializeToOstream(&output);
  output.close();
  google::protobuf::ShutdownProtobufLibrary();
}

void RNN::read(string snapshot)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  NetParamter net;
  fstream input(snapshot, ios::in | ios::binary);
  net.ParseFromIstream(&input);
  input.close();
  
  word_dim_ = net.word_dim();
  hidden_dim_ = net.hidden_dim();
  lr_ = net.learingrate();
  bptt_truncate_ = net.bptt_truncate();
  epoch_ = net.epoch();
  
  unique_ptr<double[]> U_temp(new double[word_dim_ * hidden_dim_]);
  unique_ptr<double[]> W_temp(new double[hidden_dim_ * hidden_dim_]);
  unique_ptr<double[]> V_temp(new double[word_dim_ * hidden_dim_]);
  for(int i = 0; i < word_dim_ * hidden_dim_; ++i)
    {
      U_temp[i] = net.u(i);
      V_temp[i] = net.v(i);
    }
  for(int j = 0; j < hidden_dim_ * hidden_dim_; ++j)
    W_temp[j] = net.w(j);
  U = move(U_temp);
  W = move(W_temp);
  V = move(V_temp);

  unique_ptr<double[]> dU_temp(new double [hidden_dim_ * word_dim_]);
  unique_ptr<double[]> dV_temp(new double [hidden_dim_ * word_dim_]);
  unique_ptr<double[]> dW_temp(new double [hidden_dim_ * hidden_dim_]);

  dU = move(dU_temp);
  dV = move(dV_temp);
  dW = move(dW_temp);

  fill(dU.get(), dU.get() + hidden_dim_ * word_dim_, 0.0);
  fill(dV.get(), dV.get() + hidden_dim_ * word_dim_, 0.0);
  fill(dW.get(), dW.get() + hidden_dim_ * hidden_dim_, 0.0);

  google::protobuf::ShutdownProtobufLibrary();
}
