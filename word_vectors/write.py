import struct
from word_vectors import Vocab, Vects

def _pad(word: str, max_len: int) -> bytes:
    """Pad a word out so the byte representation is max_len."""
    b = word.encode('utf-8')
    return b + b' ' * (max_len - len(b))

def write_dense(vocab: Vocab, vectors: Vects, max_len: int, file_name: str) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        max_len: The longest length of the words as bytes.
        file_name: Where to save the file.
    """
    with open(file_name, 'wb') as f:
        for val in (len(vocab), vectors.shape[1], max_len):
            f.write(struct.pack('<i', val))
        for word, vector in zip(vocab, vectors):
            word = _pad(word, max_len)
            f.write(word)
            f.write(vector.tobytes())

def write_glove(vocab: Vocab, vectors: Vects, file_name: str) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        file_name: Where to save the file.
    """
    with open(file_name, 'w') as f:
        for word, vector in zip(vocab, vectors):
            f.write(" ".join([word, *map(str, vector)]) + '\n')

def write_w2v(vocab: Vocab, vectors: Vects, file_name: str) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        file_name: Where to save the file.
    """
    with open(file_name, 'wb') as f:
        f.write(f"{len(vocab)} {vectors.shape[1]}\n".encode('utf-8'))
        for word, vector in zip(vocab, vectors):
            f.write((word + " ").encode('utf-8') + vector.tobytes())
