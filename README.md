# Word Vectors

[![PyPi Version](https://img.shields.io/pypi/v/word-vectors)](https://pypi.org/project/word-vectors/)  [![Actions Status](https://github.com/blester125/word-vectors/workflows/Unit%20Test/badge.svg)](https://github.com/blester125/word-vectors/actions) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A fast light-weight library for loading (and saving) word vectors.

## Reading

The default way to read in word vectors is to use `read`. This function will sniff the file to try to determine what
kind of file it is. If this sniffing fails for some reason an exception is raised. Don't worry you can still read your
vectors you just need to use the read function specific to the vector file format you are using.

### File Types

There are a few common types of word vectors file formats used in the NLP community. The supported formats are described
here.

#### GloVe

A simple vector file that is a plain text file. Each line is a word followed by the vectors. Each line has the word and
the elements of the vectors separated by a space. This is both slow and space inefficient.

This can be read with the `read_glove` function.

#### Word2Vec

##### Text

A text format this is the same as the GloVe format except the first line is two numbers, the first number is the number
of elements in the vocabulary and the second is the size of the vectors. These numbers are not very helpfully because
often some of these vector files have the same word at multiple lines so pre-allocating your vectors based on these
numbers doesn't really work. Like GloVe this is both slow and space inefficient.

This can be read with the `read_w2v_text` function.

##### Binary

A simple binary format where the first row is the number of items in the vocab and the size of the vectors. Each line
after is a word followed by the vector as a binary string separated by a space. This format is compact but slow because
you need to read a byte at a time to the find the end of each word.

This can be read with the `read_w2v` function.

**Note:** The popular `fastText` pretrained word vectors ship in both the text and binary formats used by word2vec.

#### Dense

This is my new format. It is a binary file where the first 12 bytes are the vocab size, vector size, and max length of
a word as unsigned, little endian, ints. Then the words and vectors follow with the words padded to the max length and
then the vector. This format is a little larger than the word2vec format but it is faster because the location of each
item (both the words or the vectors) can be calculated quickly. It also allows the possibility of multithreaded
reading. This format is smaller than the normal glove format.

This can be read with the `read_dense` function.


## Writing

Each format has its own writing function what takes in the destination file name, the vocab, and the vectors. The
available writers are the following:

 * `write_glove`
 * `write_w2v_text`
 * `write_w2v`
 * `write_dense`
