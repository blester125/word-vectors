__version__ = "1.2.0"

from typing import Dict, Tuple
from enum import Enum
import numpy as np

Vocab = Dict[str, int]
Vectors = np.ndarray

FileType = Enum("FileType", "GLOVE W2V_TEXT W2V DENSE")

INT_SIZE = 4
FLOAT_SIZE = 4
DENSE_HEADER = 3

import word_vectors.read as read_module
import word_vectors.write as write_module
from word_vectors.read import read, read_w2v, read_w2v_text, read_glove, read_dense
from word_vectors.write import write, write_w2v, write_w2v_text, write_glove, write_dense
