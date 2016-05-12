#ifndef _DATASET_H_
#define _DATASET_H_
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include "datatype.h"
#include <algorithm>
using namespace std;
template <typename DType>
class Dataset
{
 public:
  string filePath_;
  int voc_size_;
  // two hash maps for words and values correspondence
  unordered_map<string, DType> word_to_index_;
  unordered_map<DType, string> index_to_word_;


  vector <Sentence> tokenize();
  vector <vector <DType>> sentences;
  Dataset(const string &filePath, int voc_size);
  Dataset(const Dataset &data);
  void mapping();
};



#endif
