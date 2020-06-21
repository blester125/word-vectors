"""Read, Write, and Convert between different word vector serialization formats."""

__version__ = "2.1.0"

from typing import Dict, Tuple
from enum import Enum
import numpy as np


#: A mapping of word to integer index. This index is used pull the this words
#: vector from the matrix of word vectors.
Vocab = Dict[str, int]
#: The actual word vectors. These are always of rank 2 and have the shape ``[vocab size, vector size]``
Vectors = np.ndarray


class FileType(Enum):
    """An Enumeration of the Word Vector file types supported."""

    #: The format used by Glove. See :py:func:`~word_vectors.read.read_glove` for a
    #: description of file format and common pre-trained embeddings that use this format.
    GLOVE = "glove"
    #: The text format introduced by Word2Vec. See :py:func:`~word_vectors.read.read_w2v_text`
    #: for a description of the file format and common pre-trained embeddings that use this format.
    W2V_TEXT = "w2v-text"
    #: The binary format used by Word2Vec and pre-trained GoogleNews vectors. See
    #: :py:func:`~word_vectors.read.read_w2v` for a description of the file format and common
    #: pre-trained embeddings that use this format.
    W2V = "w2v"
    #: Our new Dense file format. See :py:func:`~word_vectors.read.read_dense` for a description of the file format.
    DENSE = "dense"
    #: The file format used to distribute FastText vectors, it is just the word2vec text format.
    #: See :py:func:`~word_vectors.read.read_w2v_text` for a description of the file format.
    FASTTEXT = "w2v-text"
    #: The file format used to distribute Numberbatch vectors, it is just the word2vec text format.
    #: See :py:func:`~word_vectors.read.read_w2v_text` for a description of the file format.
    NUMBERBATCH = "w2v-text"

    @classmethod
    def from_string(cls, value: str) -> "FileType":
        """Convert a string into the Enum value.

        Args:
            value: The string specifying the file type.

        Returns:
            The Enum value parsed from the string.

        Raises:
            ValueError: If the string wasn't able to be parsed into
                an Enum value.
        """
        value = value.lower()
        if value == "glove":
            return cls.GLOVE
        if value == "w2v_text" or value == "w2v-text":
            return cls.W2V_TEXT
        if value == "w2v":
            return cls.W2V
        if value == "dense":
            return cls.DENSE
        if value == "numberbatch":
            return cls.NUMBERBATCH
        if value in ("fasttext", "fast-text", "fast_text"):
            return cls.FASTTEXT
        return ValueError(f"Unable to understand file type, got: {value}")

    def __str__(self) -> str:
        """When calling ``str`` on an enum member output a value suitable for filenames"""
        return self.value


INT_SIZE = 4  #: The size of an int32 in bytes used when reading binary files.
FLOAT_SIZE = 4  #: The size of a float32 in bytes when reading a binary file.
LONG_SIZE = 8  #: The size of an int64 in bytes when reading binary files.
DENSE_HEADER = 4  #: The number of elements in the Dense format header.
DENSE_MAGIC_NUMBER = 2283  #: A magic number used to identify a Dense format file.


import word_vectors.read as read_module
import word_vectors.write as write_module
import word_vectors.convert as convert_module
from word_vectors.read import (
    read,
    read_with_vocab,
    read_w2v,
    read_w2v_with_vocab,
    read_w2v_text,
    read_w2v_with_vocab,
    read_glove,
    read_glove_with_vocab,
    read_dense,
    read_dense_with_vocab,
    verify_dense,
)
from word_vectors.convert import (
    convert,
    w2v_to_dense,
    w2v_to_glove,
    w2v_to_w2v_text,
    glove_to_w2v,
    glove_to_w2v_text,
    glove_to_dense,
    w2v_text_to_w2v,
    w2v_text_to_glove,
    w2v_text_to_dense,
    dense_to_glove,
    dense_to_w2v,
    dense_to_w2v_text,
)
from word_vectors.write import write, write_w2v, write_w2v_text, write_glove, write_dense
