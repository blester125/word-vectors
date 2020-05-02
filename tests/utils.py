import random
import string
from pathlib import Path
import numpy as np
from word_vectors.write import write_dense

DATA = Path(__file__).parent / "test_data"
GLOVE = "glove.txt"
W2V = "w2v.bin"
W2V_TEXT = "w2v.txt"
DENSE = "dense.bin"
DIM = 20
VOCAB = 15
vocab = {str(i): i for i in range(VOCAB)}
MAX_VOCAB = 2
vectors = np.zeros((VOCAB, DIM), dtype=np.float32)
scale = np.reshape(np.arange(VOCAB, dtype=np.float32), (-1, 1))
vectors = vectors + scale

GLOVE_DUPPED = "dupped.glove"
W2V_DUPPED = "dupped.w2v"
DENSE_DUPPED = "dupped.dense"
W2V_TEXT_DUPPED = "dupped.w2v.txt"
dupped_vocab = {"a": 0, "b": 1, "c": 2}
dupped_vectors = np.arange(4 * 20).reshape(4, -1)[[0, 1, 3]]


def rand_str(length: int = None, min_: int = 3, max_: int = 5):
    length = random.randint(min_, max_) if length is None else length
    return "".join(random.choice(string.ascii_letters) for _ in range(length))
