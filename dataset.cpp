#include "dataset.h"
template <typename DType>
Dataset<DType>::Dataset(const string &filePath, int voc_size)
{
  filePath_ = filePath;
  voc_size_ = voc_size;
}

template <typename DType>
vector <Sentence> Dataset<DType>::tokenize()
{
  // however, this function can only be dataset specific
  // it can not be reused with other datasets
  fstream dataFile;
  dataFile.open(filePath_, ios::in);
  string word;

  // result: stores the dataset sentence by sentence
  vector <Sentence> result;

  // repeated_list: stores all words appeared in the dataset
  // with their numbers of occurences
  map<string, int> dictionary;
  Sentence sentence;


  while(!dataFile.eof())
    {
      dataFile >> word;
      if (word == "Book")
	{
	  string line;
	  getline(dataFile, line);
	  continue;
	}
      char terminator = word.back();
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      // judge whether it is the last word of a sentence
      if (terminator != '.' && terminator != ';' && terminator != '?')
	{
	  // the word is a normal word
	  if ( (terminator >= 97 && terminator <= 122) ||
	       (terminator >= 65 && terminator <= 90))
	    {
	      sentence.add_word(word);
	      dictionary[word] += 1;
	    }
	  else if (terminator >= '0' && terminator <= '9')
	    {
	      continue;
	    }
	  else
	    {
	      sentence.add_word(word.substr(0, word.length() - 1));
	      dictionary[word.substr(0, word.length() - 1)] += 1;
	    }
	}
      else
	{
	  sentence.add_word(word.substr(0, word.length() - 1));
	  dictionary[word.substr(0, word.length() - 1)] += 1;
	  result.push_back(sentence);

	  sentence.clear();
	}

    }


  // select voc_size_ most frequently used words
  // to form our vocabulary
  vector <pair<string ,int>> dict;
  for(auto it = dictionary.begin(); it != dictionary.end(); ++it)
    {
      dict.push_back(*it);
    }
  auto cmp = [](pair<string, int> const &a, pair <string, int> const &b)
    {
      return a.second >= b.second;
    };
  sort(dict.begin(), dict.end(), cmp);

  for(size_t i = voc_size_; i < dict.size(); ++i)
    dictionary.erase(dict[i].first);

  // dictionary: stores the most frequently used voc_size_ words
  // we remap them to DType (without word2vec)
  int val = 3;
  for(auto it = dictionary.begin(); it != dictionary.end(); ++it)
    {
      word_to_index_[it -> first] = static_cast<DType>(val);
      index_to_word_[static_cast<DType>(val)] = it -> first;
      val++;
    }

  dataFile.close();

  return result;
}

template <typename DType>
void Dataset<DType>::mapping()
{
  vector <Sentence> sentence_list = tokenize();

  sentences.resize(sentence_list.size());

  // we map source token to 0
  // sink token to 1
  // unknown token to 2
  for(size_t i = 0; i < sentence_list.size(); ++i)
    {
      sentences[i].push_back(static_cast<DType>(0));
      for (size_t j = 0; j < sentence_list[i].words_.size(); ++j)
	{
	  DType val = word_to_index_[sentence_list[i].words_[j]];

	  if(static_cast<int>(val) == 0)
	    val = static_cast<DType>(2);
	  sentences[i].push_back(val);
	}
      sentences[i].push_back(static_cast<DType>(1));
    }

  // for(size_t j = 0; j < 10; ++j)
  //   {
  //     for(size_t i = 0; i < sentences[j].size(); ++i)
  // 	cout << sentences[j][i] << " ";
  //     cout << endl;
  //   }
}                       
