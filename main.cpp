#include "dataset.h"
#include "rnn.hpp"
#include "dataset.cpp"

using namespace std;
RNN<float> *rnn;

static void interrupt(int signum)
{
  cout << "Saving model parameters to rnn_model.snapshot" << endl;
  rnn -> write("rnn_model.snapshot");
  delete rnn;
  exit(1);
}

int main()
{
  cout.setf(ios::fixed);
  cout << setprecision(10);

  // // gradient check
  // int grad_check_vocab_size = 250;
  // RNN<double> rnn_check(grad_check_vocab_size, 10, 1000);
  // mode md = GPU;
  // rnn_check.set_mode(md);
  // rnn_check.initialize();
  // vector <int> xx{0,1,2,3,4};
  // vector <int> yy{1,2,3,4,5};
  // rnn_check.gradient_check_gpu(xx,yy);

  mode md = GPU;
  rnn = new RNN<float>(4000,100,1000);
  //rnn = new RNN("rnn_model.snapshot");
  rnn -> set_mode(md);
  rnn -> initialize();
  string dataPath = "../kjv12/KJV12.TXT";
  Dataset<int> dataset(dataPath, 4000);

  dataset.mapping();

  size_t total_size = dataset.sentences.size();
  cout << "dataset size = " << total_size << endl;

  vector <int> val_index;
  vector <int> train_index;
  fstream train("../kjv12/train.txt", ios::in);
  fstream val("../kjv12/test.txt", ios::in);
  for(int i = 0; i < 1000; ++i)
    {
      int val_ind;
      val >> val_ind;
      val_index.push_back(val_ind);
    }
  for(size_t i = 0; i < total_size - 1000; ++i)
    {
      int train_ind;
      train >> train_ind;
      train_index.push_back(train_ind);
    }


  vector <vector <int>> X_train;
  vector <vector <int>> y_train;
  vector <vector <int>> x_val;
  vector <vector <int>> y_val;
  X_train.resize(total_size - 1000);
  y_train.resize(total_size - 1000);
  x_val.resize(1000);
  y_val.resize(1000);

  for(size_t i = 0; i < 1000; ++i)
    {
      x_val[i].resize(dataset.sentences[val_index[i]].size() - 1);
      y_val[i].resize(dataset.sentences[val_index[i]].size() - 1);
      copy(dataset.sentences[val_index[i]].begin(), dataset.sentences[val_index[i]].end() - 1,
	   x_val[i].begin());
      copy(dataset.sentences[val_index[i]].begin() + 1, dataset.sentences[val_index[i]].end(),
	   y_val[i].begin());

    }
  for(size_t i = 0; i < total_size - 1000; ++i)
    {
      int ind = train_index[i];
      X_train[i].resize(dataset.sentences[ind].size() - 1);
      y_train[i].resize(dataset.sentences[ind].size() - 1);
      copy(dataset.sentences[ind].begin(), dataset.sentences[ind].end() - 1, X_train[i].begin());
      copy(dataset.sentences[ind].begin() + 1, dataset.sentences[ind].end(), y_train[i].begin());
    }


  signal(SIGINT, interrupt);
  rnn -> train_gpu(X_train, y_train, x_val, y_val, 0.005, 1000, 5, 9, 15);
  delete rnn;
  return 0;
}
