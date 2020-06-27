------------
Word Vectors
------------


.. image:: https://img.shields.io/pypi/v/word-vectors
    :target: https://pypi.org/project/word-vectors/
    :alt: PyPI Version
.. image:: https://github.com/blester125/word-vectors/workflows/Unit%20Test/badge.svg
    :target: https://github.com/blester125/word-vectors/actions
    :alt: Actions Status
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black
.. image:: https://readthedocs.org/projects/word-vectors/badge/?version=latest
    :target: https://word-vectors.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

A fast, light-weight library for reading, writing, and converting between various word vector serialization formats.

.. contents::
   :local:
   :depth: 2

What are Word Vectors?
======================

Word vectors are low-dimensional, dense representations of words. This sounds very complicated but then you boil it down
is becomes a lot clearer. The it really means that each word is associated with a list of numbers (a vector) that are
used to represent the semantic meaning of that word. There vectors normal range in size from as little as 100 elements
to around 300. It might seem like a stretch to call that "low-dimensional" but these vectors are very small compared to
older methods of vector representations of words. Words used to be encoded as "one-hot" vectors where each word was
given a unique index and the vector was full or zeros except for a one at that index. This results in massive vectors
(each vector is the size of the vocabulary and the vector size scales linearly as the vocabulary grows). The other
problem with this method is that vectors are orthogonal. All none word index elements are zero so when you do something
like a dot product between two vectors you will always get zero. Dense vectors, on the other hand, have a fixed size
(as you add more terms to your vocabulary the vectors stay the same size) and when you take the dot product of two
vectors you get non-zero values. This can be used for tasks like semantic similarity between different words. For a more
complete introduction to word vectors and the algorithms used to crate them check out these lectures from
Stanford.

 - `Word Vectors`_
 - `Word Vectors and Word Senses`_

.. _Word Vectors: https://www.youtube.com/watch?v=8rXD5-xhemo
.. _Word Vectors and Word Senses: https://www.youtube.com/watch?v=kEMJRjEdNzM

Supported File Formats
======================

This library supports reading and writing several formats of vector serialization. These formats are often
under-specified and only truly defined by the implementations of the original software than wrote out the vectors. In
the next section we quickly summarize some of the most common file formats.

GloVe
-----

The GloVe format is a pure text format. Each (word, vector) pair is represented
by a single line in the file. The line starts with the word, a space, and then
the float32 text representations of the elements in the vector associated with
that word. Each of these vector elements are also separated with a space.

The main vectors distributed in this format are the `GloVe`_ vectors
`(Pennington, et. al., 2014)`_

.. _GloVe: https://nlp.stanford.edu/projects/glove/
.. _(Pennington, et. al., 2014): https://www.aclweb.org/anthology/D14-1162/

Word2Vec
--------

There are two different vector serialization file formats introduced by the
`word2vec software`_ `(Mikolov, et. al., 2013)`_. One is a pure text format
and the other a binary one.

.. _word2vec software: https://code.google.com/archive/p/word2vec/
.. _(Mikolov, et. al., 2013): https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality

Text
~~~~

The word2vec text format is a pure text format. The first line is two integers, represented as text and separated by a
space, that specify the number of types in the vocabulary and the size of the word vectors respectively. Each following
line represents a (word, vector) pair. The line stars with the word, a space, and then the float 32 text representations
of the elements in the vector associated with that word. Each of these vector elements are also separated with a space.

One can see that that this is actually the same as the GloVe format except that in GloVe they removed the header line.

The main embeddings distributed in this format are `FastText`_ `(Bojanowski, et. al., 2017)`_ and `NumberBatch`_ `(Speer, et. al., 2017)`_

.. _FastText: https://fasttext.cc/
.. _(Bojanowski, et. al., 2017): https://www.aclweb.org/anthology/Q17-1010/
.. _NumberBatch: https://github.com/commonsense/conceptnet-numberbatch
.. _(Speer, et. al., 2017): https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972


Binary
~~~~~~

The word2vec binary format is a mix of textual an binary representations.
The first line is two integers (as text, separated be a space) representing
the number of types in the vocabulary and the size of the word vectors
respectively. (word, vector) pairs follow. The word is represented as text
and a space. After the space each element of a vector is represented as a
binary float32.

The most well-known pre-trained embeddings distributed in this format are
the `GoogleNews`_ vectors.

.. DANGER::

    There is no formal definition of this file format, the only definitive
    reference on it is the original implementation in the `word2vec software`_

    Due to the lack of a definition (and no special handling of it in the code)
    there is no explicit statements about the endianess of the binary representations.
    Most code just uses the ``numpy.from_buffer`` and that seems to work now that
    most people have little-endian machines. However due to the lack of explicit
    direction on this encoding I would advise caution when loading vectors that
    were trained on big-endian hardware.

.. _word2vec software: https://code.google.com/archive/p/word2vec/
.. _(Mikolov, et. al., 2013): https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality
.. _GoogleNews: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Leader
------

This is our fully binary vector format.

The first line is a header for the leader format and it is a 3-tuple.
The elements of this tuple are: A magic number, the size of the vocabulary,
and the size of the vectors. These numbers are represented as little-endian
unsigned long longs that have a size of 8 bytes.

Following the header there are (length, word, vector) tuples. The length is
the length of this particular word encoded as a little-endian unsigned
integer. The word is stored as ``utf-8`` bytes. After the word the vector
is stored where each element is a little-endian float32 (4 bytes).

By tracking the length of the word we can jump directly to the start of them
vector instead of having to iterate through the word like we do in the word2vec
binary format.

.. NOTE::

    The magic number if used to make sure this is can actual
    file and not just trying to extract word vectors from a
    random binary file. The Magic Number is ``38941``.

.. NOTE::

    One of the downsides of this format is that it is harder
    to inspect the file to see information like the vocabulary
    size or the vector size. Unlink the Word2Vec format the
    header is not text so a simple ``head -n 1 embedding-file``
    will **NOT** work. Instead you can use
    ``od -l --endian=little -N 24 embedding-file`` and you should
    see the magic number, the vocabulary size, the vector size, and
    the max length of the tokens (as ``utf-8`` bytes).

`A note on the Senna format`: There is an older format of embeddings called `Senna embeddings`_ `(Collobert, et. al.,
2011)`_. The format actually uses two files. There is a vocabulary file where each line has a single word and an
vector file where each line has the text representations of the float32 elements in a vector separated by a
space. These files are aligned so that the word on line ``i`` of the word file is represented by the vector on line
``i`` of the vector file. Due to the mismatch in API supporting this format would cause (requiring two file
rather than just one) we have decided not to provide reading utilities for this format. Luckily the conversion of this
format into the GloVe format is a single ``paste`` command.

.. code:: bash

    paste -d" " /path/to/word/file.senna /path/to/vector/file.senna > word_vectors.glove

.. _Senna embeddings: https://ronan.collobert.com/senna/
.. _(Collobert, et. al., 2011): http://ronan.collobert.com/pub/matos/2011_nlp_jmlr.pdf

Usage
=====

While these vector formats are not very complex it is annoying to have to write code to read them in for each
project. This causes a lot of people to pull in pretty large libraries just to use the vector reading functionality.
The problem with this (beside the heavy dependency) is that these libraries tend to return the vocabulary and vectors
within some complex, library specific class. There is often a lot of utility to be gained from these classes when you
are actually using the rest of the library but when all you care about is reading in the vectors this is a hindrance.

We designed this library to fix both of these at once. The library is small and focused. You won't be pulling in a lot
of code that does (really cool) things you will never touch. We also return results using the simplest formats possible
for maximum flexibility.

The main data structure that people conceptually think about when working with word vectors is a mapping for word to
vector. This is natural to represent as a python dictionary. This isn't the format that people actually use
however. Having many single vectors inside of a dictionary is less space efficient and harder to work with than a single
large matrix the vectors stacked on one another. When using this format the data structure that comes to mind is an pair
of associated arrays. The word at index ``i`` in one array is associated with the vector at index ``i`` in the
other. The main use case is a look up from word to vector however so instead of storing an actual list of words we use a
dictionary mapping words to integers. These integers can then be used to look up the vector in the dense matrix.

Our vocabulary is simply ``Dict[str, int]`` and our vectors type is just a ``np.ndarray`` of size
``[number of words in vocab, size of vector]``.

These simple datatypes give us a lot of flexibility downstream. First we read in the vocabulary and vectors from a file.

.. code:: python

    >>> from word_vectors import read
    >>> v, wv = read("/home/blester/embeddings/glove-6B.100d")
    >>> len(v)
    400000
    >>> wv.shape
    (400000, 50)

Then we can lookup a single word by getting its index in the vocabulary and pulling the vector from the matrix.

.. code:: python

    >>> wv[v['the']]
    array([ 4.1800e-01,  2.4968e-01, -4.1242e-01,  1.2170e-01,  3.4527e-01,
           -4.4457e-02, -4.9688e-01, -1.7862e-01, -6.6023e-04, -6.5660e-01,
            2.7843e-01, -1.4767e-01, -5.5677e-01,  1.4658e-01, -9.5095e-03,
            1.1658e-02,  1.0204e-01, -1.2792e-01, -8.4430e-01, -1.2181e-01,
           -1.6801e-02, -3.3279e-01, -1.5520e-01, -2.3131e-01, -1.9181e-01,
           -1.8823e+00, -7.6746e-01,  9.9051e-02, -4.2125e-01, -1.9526e-01,
            4.0071e+00, -1.8594e-01, -5.2287e-01, -3.1681e-01,  5.9213e-04,
            7.4449e-03,  1.7778e-01, -1.5897e-01,  1.2041e-02, -5.4223e-02,
           -2.9871e-01, -1.5749e-01, -3.4758e-01, -4.5637e-02, -4.4251e-01,
            1.8785e-01,  2.7849e-03, -1.8411e-01, -1.1514e-01, -7.8581e-01],
           dtype=float32)
    >>> wv[v['the']].shape
    (50,)

We can also lookup an entire sentence in a single go getting back a dense matrix of ``[tokens, embeddings]`` which is
perfect for downstream machine leaning applications like the input to neural networks.

.. code:: python

    >>> wv[[v[t] for t in "the quick brown fox".split()]]
    array([[ 4.1800e-01,  2.4968e-01, -4.1242e-01,  1.2170e-01,  3.4527e-01,
            -4.4457e-02, -4.9688e-01, -1.7862e-01, -6.6023e-04, -6.5660e-01,
             2.7843e-01, -1.4767e-01, -5.5677e-01,  1.4658e-01, -9.5095e-03,
             1.1658e-02,  1.0204e-01, -1.2792e-01, -8.4430e-01, -1.2181e-01,
            -1.6801e-02, -3.3279e-01, -1.5520e-01, -2.3131e-01, -1.9181e-01,
            -1.8823e+00, -7.6746e-01,  9.9051e-02, -4.2125e-01, -1.9526e-01,
             4.0071e+00, -1.8594e-01, -5.2287e-01, -3.1681e-01,  5.9213e-04,
             7.4449e-03,  1.7778e-01, -1.5897e-01,  1.2041e-02, -5.4223e-02,
            -2.9871e-01, -1.5749e-01, -3.4758e-01, -4.5637e-02, -4.4251e-01,
             1.8785e-01,  2.7849e-03, -1.8411e-01, -1.1514e-01, -7.8581e-01],
           [ 1.3967e-01, -5.3798e-01, -1.8047e-01, -2.5142e-01,  1.6203e-01,
            -1.3868e-01, -2.4637e-01,  7.5111e-01,  2.7264e-01,  6.1035e-01,
            -8.2548e-01,  3.8647e-02, -3.2361e-01,  3.0373e-01, -1.4598e-01,
            -2.3551e-01,  3.9267e-01, -1.1287e+00, -2.3636e-01, -1.0629e+00,
             4.6277e-02,  2.9143e-01, -2.5819e-01, -9.4902e-02,  7.9478e-01,
            -1.2095e+00, -1.0390e-02, -9.2086e-02,  8.4322e-01, -1.1061e-01,
             3.0096e+00,  5.1652e-01, -7.6986e-01,  5.1074e-01,  3.7508e-01,
             1.2156e-01,  8.2794e-02,  4.3605e-01, -1.5840e-01, -6.1048e-01,
             3.5006e-01,  5.2465e-01, -5.1747e-01,  3.4705e-03,  7.3625e-01,
             1.6252e-01,  8.5279e-01,  8.5268e-01,  5.7892e-01,  6.4483e-01],
           [-8.8497e-01,  7.1685e-01, -4.0379e-01, -1.0698e-01,  8.1457e-01,
             1.0258e+00, -1.2698e+00, -4.9382e-01, -2.7839e-01, -9.2251e-01,
            -4.9409e-01,  7.8942e-01, -2.0066e-01, -5.7371e-02,  6.0682e-02,
             3.0746e-01,  1.3441e-01, -4.9376e-01, -5.4788e-01, -8.1912e-01,
            -4.5394e-01,  5.2098e-01,  1.0325e+00, -8.5840e-01, -6.5848e-01,
            -1.2736e+00,  2.3616e-01,  1.0486e+00,  1.8442e-01, -3.9010e-01,
             2.1385e+00, -4.5301e-01, -1.6911e-01, -4.6737e-01,  1.5938e-01,
            -9.5071e-02, -2.6512e-01, -5.6479e-02,  6.3849e-01, -1.0494e+00,
             3.7507e-02,  7.6434e-01, -6.4120e-01, -5.9594e-01,  4.6589e-01,
             3.1494e-01, -3.4072e-01, -5.9167e-01, -3.1057e-01,  7.3274e-01],
           [ 4.4206e-01,  5.9552e-02,  1.5861e-01,  9.2777e-01,  1.8760e-01,
             2.4256e-01, -1.5930e+00, -7.9847e-01, -3.4099e-01, -2.4021e-01,
            -3.2756e-01,  4.3639e-01, -1.1057e-01,  5.0472e-01,  4.3853e-01,
             1.9738e-01, -1.4980e-01, -4.6979e-02, -8.3286e-01,  3.9878e-01,
             6.2174e-02,  2.8803e-01,  7.9134e-01,  3.1798e-01, -2.1933e-01,
            -1.1015e+00, -8.0309e-02,  3.9122e-01,  1.9503e-01, -5.9360e-01,
             1.7921e+00,  3.8260e-01, -3.0509e-01, -5.8686e-01, -7.6935e-01,
            -6.1914e-01, -6.1771e-01, -6.8484e-01, -6.7919e-01, -7.4626e-01,
            -3.6646e-02,  7.8251e-01, -1.0072e+00, -5.9057e-01, -7.8490e-01,
            -3.9113e-01, -4.9727e-01, -4.2830e-01, -1.5204e-01,  1.5064e+00]],
            dtype=float32)
    >>> wv[[v[t] for t in "the quick brown fox".split()]].shape
    (4, 50)

Reading
-------

Reading is most often done with the ``word_vectors.read.read`` function. We can use the
``word_vectors.FileType`` argument to specify a specific format to read the file as or we can let the code
infer the format for itself (you can also use one of the format specific readers to read a certain file format. The read
API is very simply just pass in the file name.

.. code:: python

    >>> from word_vectors.read import read
    >>> # Read where the format is determined by sniffing
    ... w, wv = read("/path/to/vector-file")
    >>> from word_vectors import FileType
    >>> # Read using the binary Word2Vec format
    ... v, wv = read("/path/to/vector-file", FileType.W2V)
    >>> from word_vectors.read import read_leader
    >>> # Read leader formatted vectors
    ... v, wv = read_leader("/path/to/leader-vector-file")


You can also use the ``_with_vocab`` version of all the reader function to only read a subsection of the
vocabulary. Below we can see an example. First we read the full vocabulary from the file. We can see that is
has the string representations of numbers from zero for fourteen. We can see the vectors for several tokens.
Then we create a user vocabulary that only has the even numbers, and we re-read the vectors with this vocab.
We see that we have now only read in a subset of the word and that our vocab is in the same order that we
passed in. We can also see the vectors for a word haven't changed. Finally we re-read the vectors again but
this time we ask for it to keep the vectors in the pre-train vocabulary that are not present in our vocab using
``keep_extra=True``. We can see the indices from our user vocabulary have not changed but we get the full
vocabulary back with the extra words appearing at the end.

.. code:: python

    >>> from word_vectors import read, read_with_vocab
    >>> v, wv = read("leader.bin")
    >>> v
    {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14}
    >>> wv[v["4"]]
    array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
           4., 4., 4.], dtype=float32)
    >>> wv[v["13"]]
    array([13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13.,
           13., 13., 13., 13., 13., 13., 13.], dtype=float32)
    >>> wv.shape
    (15, 20)
    >>> user_vocab = {k: i for i, k in enumerate(k for k, x in v.items() if x % 2 == 0)}
    >>> user_vocab
    {'0': 0, '2': 1, '4': 2, '6': 3, '8': 4, '10': 5, '12': 6, '14': 7}
    >>> v, wv = read_with_vocab("leader.bin", user_vocab)
    >>> v
    {'0': 0, '2': 1, '4': 2, '6': 3, '8': 4, '10': 5, '12': 6, '14': 7}
    >>> wv[v["4"]]
    array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
           4., 4., 4.], dtype=float32)
    >>> wv.shape
    (8, 20)
    >>> v, wv = read_with_vocab("leader.bin", user_vocab, keep_extra=True)
    >>> v
    {'0': 0, '2': 1, '4': 2, '6': 3, '8': 4, '10': 5, '12': 6, '14': 7, '1': 8, '3': 9, '5': 10, '7': 11, '9': 12, '11': 13, '13': 14}
    >>> wv[v["4"]]
    array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
           4., 4., 4.], dtype=float32)
    >>> wv[v["13"]]
    array([13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13.,
           13., 13., 13., 13., 13., 13., 13.], dtype=float32)
    >>> wv.shape
    (15, 20)


Writing
-------

Writing similarly has a main ``word_vectors.write.write`` function that dispatches on the
``word_vectors.FileType`` argument and there are format specific writers if you want to use those instead.

.. code:: python

    >>> from word_vectors.read import read
    >>> v, wv = read("/path/to/vectors")
    >>> from word_vectors import FileType
    >>> from word_vectors.write import write
    >>> write("/path/to/vectors.leader", v, wv, FileType.LEADER)
    >>> write("/path/to/vectors.w2v", v, wv, FileType.W2V)
    >>> write_glove("/path/to/vectors.glove", v, wv)

Converting
----------

Conversions also have a general function (``word_vectors.convert.convert``) dispatching on
``word_vectors.FileType`` and specific functions for converting between certain pairs.

.. code:: python

    >>> from word_vectors import FileType
    >>> from word_vectors.convert import convert
    >>> # Conversion to w2v via sniffing the original file
    ... convert("/path/to/vectors", output="/path/to/vectors.w2v", output_file_type=FileType.W2V)
    >>> # Conversion to w2v with an explicit input type
    ... convert(
    ...     "/path/to/vectors.glove",
    ...     output="/path/to/vectors.w2v",
    ...     output_file_type=FileType.w2v,
    ...     input_file_type=FileType.GLOVE
    ... )
    >>> # Converting between specific formats
    >>> from word_vectors.convert import w2v_text_to_w2v
    ... w2v_text_to_w2v("/path/to/vectors.w2v-text", output="/path/to/vectors.w2v")
