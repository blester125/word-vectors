from pathlib import Path
import numpy as np
from word_vectors.write import write_dense

DATA = Path(__file__).parent / "test_data"
GLOVE = "glove.txt"
W2V = "w2v.bin"
DENSE = "dense.bin"
DIM = 20
VOCAB = 15
vocab = {str(i): i for i in range(VOCAB)}
MAX_VOCAB = 2
vectors = np.zeros((VOCAB, DIM), dtype=np.float32)
scale = np.reshape(np.arange(VOCAB, dtype=np.float32), (-1, 1))
vectors = vectors + scale


if __name__ == "__main__":
    main()
