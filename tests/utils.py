from pathlib import Path
import numpy as np
from word_vectors.write import write_dense

DATA = Path(__file__).parent / "test_data"
GLOVE = 'glove.txt'
W2V = 'w2v.bin'
DENSE = 'dense.bin'
DIM = 20
VOCAB = 15
vocab = {str(i): i for i in range(VOCAB)}
vectors = np.zeros((VOCAB, DIM), dtype=np.float32)
scale = np.reshape(np.arange(VOCAB, dtype=np.float32), (-1, 1))
vectors = vectors + scale


def main():
    with open(DATA / GLOVE, 'w') as f:
        for word, vector in zip(vocab, vectors):
            f.write(" ".join([word, *map(str, vector)]) + "\n")

    with open(DATA / W2V, 'wb') as f:
        f.write(f"{len(vocab)} {dim}\n".encode('utf-8'))
        for word, vector in zip(vocab, vectors):
            f.write((word + " ").encode('utf-8') + vector.tobytes())

    write_dense(vocab, vectors, 2, DATA / DENSE)


if __name__ == '__main__':
    main()
