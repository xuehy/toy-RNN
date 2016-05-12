#include "datatype.h"

ostream& operator << (ostream& stream, const Sentence &sentence)
{
  for (auto &w : sentence.words_)
    stream << w << " ";
  return stream;
}

void Sentence::clear()
{
  words_.clear();
}

void Sentence::add_word(string &word)
{
  words_.push_back(word);
}

void Sentence::add_word(string &&word)
{
  words_.push_back(word);
}                       
