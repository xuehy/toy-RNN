#ifndef _DATATYPE_H_
#define _DATATYPE_H_

#include <vector>
#include <iostream>

using namespace std;

class Sentence
{
 public:
  Sentence(){};
  void clear();
  void add_word(string &word);
  void add_word(string &&word);
  vector <string> words_;
  friend ostream& operator <<(ostream& stream, const Sentence &sentence);
};


#endif
