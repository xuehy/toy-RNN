#include "dataset.h"
#include "rnn.hpp"
#include "dataset.cpp"
#include <cublas_v2.h>
using namespace std;

int main()
{
  string dataPath = "/home/xuehy/Dataset/kjv12/KJV12.TXT";
  Dataset<int> dataset(dataPath, 4000);

  dataset.mapping();

  RNN rnn(4000, 100, 1000);

  size_t total_size = 2000; //dataset.sentences.size();

  cout << "dataset size = " << total_size << endl;
  size_t val = 200;

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(0, total_size - 1);
  
  vector <vector <int>> X_train;
  vector <vector <int>> y_train;
  vector <vector <int>> x_val;
  vector <vector <int>> y_val;
  X_train.resize(total_size - val);
  y_train.resize(total_size - val);
  x_val.resize(val);
  y_val.resize(val);

  vector <int> val_index;
  val_index.resize(val);
  for(size_t i = 0; i < val; ++i)
    val_index[i] = 9 * i + 10;//100 * i + 34;
  
  for(size_t i = 0; i < val; ++i)
    {
      x_val[i].resize(dataset.sentences[val_index[i]].size() - 1);
      y_val[i].resize(dataset.sentences[val_index[i]].size() - 1);
      copy(dataset.sentences[val_index[i]].begin(), dataset.sentences[val_index[i]].end() - 1,
	   x_val[i].begin());
      copy(dataset.sentences[val_index[i]].begin() + 1, dataset.sentences[val_index[i]].end(),
	   y_val[i].begin());

    }
  for(size_t i = 0, ind = 0; i < total_size; ++i)
    {
      bool flag = false;
      for(size_t s = 0; s < val; ++s)
	{
	  if(val_index[s] == i)
	    {
	      flag = true;
	      break;
	    }
	}
      if (flag) continue;
      X_train[ind].resize(dataset.sentences[i].size() - 1);
      y_train[ind].resize(dataset.sentences[i].size() - 1);
      copy(dataset.sentences[i].begin(), dataset.sentences[i].end() - 1, X_train[ind].begin());
      copy(dataset.sentences[i].begin() + 1, dataset.sentences[i].end(), y_train[ind].begin());
      ind++;
    }

  cout.setf(ios::fixed);

  cout << setprecision(10);

  int grad_check_vocab_size = 100;
  RNN rnn_check(grad_check_vocab_size, 10, 1000);
  vector <int> x{0,1,2,3};
  vector <int> y{1,2,3,4};


  rnn_check.gradient_check(x,y);

  rnn.train(X_train, y_train, x_val, y_val, 0.005, 100, 2, 5);
  return 0;
}
