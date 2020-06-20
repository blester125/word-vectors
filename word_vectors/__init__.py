"""Read, Write, and Convert between different word vector serialization formats."""

__version__ = "1.3.0"

from typing import Dict, Tuple
from enum import Enum
import numpy as np


Vocab = Dict[str, int]  #: The mapping of word to integers we return. The int is used to map from the word into the vectors.
Vectors = np.ndarray  #: The type of the vectors we return. These are always of rank 2 and have the same ``[vocab size, vector size]``


class FileType(Enum):
    """An Enumeration of the Word Vector file types supported."""

    GLOVE = 1  #: The format used by Glove
    W2V_TEXT = 2
    W2V = 3
    DENSE = 4
    FASTTEXT = 2
    NUMBERBATCH = 2

    @classmethod
    def from_string(cls, value: str) -> 'FileType':
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


INT_SIZE = 4  #: The size of an int32 in bytes used when reading binary files.
FLOAT_SIZE = 4  #: The size of a float32 in bytes when reading a binary file.
LONG_SIZE = 8  #: The size of an int64 in bytes when reading binary files.
DENSE_HEADER = 4  #: The number of elements in the Dense format header.
DENSE_MAGIC_NUMBER = 2283  #: A magic number used to identify a Dense format file.


import word_vectors.read as read_module
import word_vectors.write as write_module
import word_vectors.convert as convert_module
from word_vectors.read import read, read_w2v, read_w2v_text, read_glove, read_dense, verify_dense
from word_vectors.convert import convert
from word_vectors.write import write, write_w2v, write_w2v_text, write_glove, write_dense
