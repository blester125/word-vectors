# Word Vectors

[![Build Status](https://travis-ci.com/blester125/word_vectors.svg?branch=master)](https://travis-ci.com/blester125/word_vectors)

A fast light library for loading word vectors.

### File Types

#### Glove

A simple vector file that is a plain text file. Each line is a word followed by the vectors with each component (and the word) separated by a space.

This is both slow and space efficient.

#### Word2Vec

A simple binary format where the first row is the number of items in the vocab and the size of the vectors. On the next line is a word followed by the vector as a binary string separated by a space.

This format is compact but slow because you need to read a byte at a time to the find the end of each word.

#### Dense

This is the new format. It is a binary file where the first 12 bytes are the vocab size, vector size, and max length of a word as unsigned, little endian, ints. Then the words and vectors follow with the words padded to the max length and then the vector.

This format is a little larger than the word2vec format but it is faster because the location of each item can be calculated quickly. It also allows the possibility of multithreaded reading. This format is smaller than the normal glove format.
