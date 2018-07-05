__version__ = '0.1.2'

from typing import Dict, Tuple
import numpy as np

Vocab = Dict[str, int]
Vects = np.ndarray

INT_SIZE = 4
FLOAT_SIZE = 4
DENSE_HEADER = 3

from word_vectors.read import read, read_w2v, read_glove, read_dense
from word_vectors.write import write_w2v, write_glove, write_dense
