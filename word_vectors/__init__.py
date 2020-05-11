__version__ = "1.3.0"

from typing import Dict, Tuple
from enum import Enum, auto
import numpy as np

Vocab = Dict[str, int]
Vectors = np.ndarray


class FileType(Enum):
    GLOVE = auto()
    W2V_TEXT = auto()
    W2V = auto()
    DENSE = auto()

    @classmethod
    def from_string(cls, value):
        value = value.lower()
        if value == "glove":
            return cls.GLOVE
        if value == "w2v_text" or value == "w2v-text":
            return cls.W2V_TEXT
        if value == "w2v":
            return cls.W2V
        if value == "dense":
            return cls.DENSE
        return ValueError(f"Unable to understand file type, got: {value}")


INT_SIZE = 4
FLOAT_SIZE = 4
DENSE_HEADER = 3

import word_vectors.read as read_module
import word_vectors.write as write_module
from word_vectors.read import read, read_w2v, read_w2v_text, read_glove, read_dense
from word_vectors.write import write, write_w2v, write_w2v_text, write_glove, write_dense
