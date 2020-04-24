import struct
from operator import itemgetter
from typing import Union, TextIO, BinaryIO, Optional, List
from file_or_name import file_or_name
from word_vectors import Vocab, Vectors
from word_vectors.utils import _find_max
from word_vectors.utils import _find_max


def _pad(word: str, max_len: int) -> bytes:
    """Pad a word out so the byte representation is max_len."""
    b = word.encode("utf-8")
    return b + b" " * (max_len - len(b))


def vocab_to_list(vocab: Vocab) -> List[str]:
    return [k for k, _ in sorted(vocab.items(), key=itemgetter(1))]


@file_or_name(wf="wb")
def write_dense(
    wf: Union[str, BinaryIO], vocab: Union[Vocab, List[str]], vectors: Vectors, max_len: Optional[int] = None
) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        max_len: The longest length of the words as bytes.
        file_name: Where to save the file.
    """
    vocab = vocab_to_list(vocab) if isinstance(vocab, dict) else vocab
    max_len = _find_max(vocab) if max_len is None else max_len
    for val in (len(vocab), vectors.shape[1], max_len):
        wf.write(struct.pack("<i", val))
    for word, vector in zip(vocab, vectors):
        word = _pad(word, max_len)
        wf.write(word)
        wf.write(vector.tobytes())


@file_or_name(wf="w")
def write_glove(wf: Union[str, TextIO], vocab: Vocab, vectors: Vectors) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        file_name: Where to save the file.
    """
    vcab = vocab_to_list(vocab) if isinstance(vocab, dict) else vocab
    for word, vector in zip(vocab, vectors):
        wf.write(" ".join([word, *map(str, vector)]) + "\n")


@file_or_name(wf="wb")
def write_w2v(wf: Union[str, BinaryIO], vocab: Vocab, vectors: Vectors) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        file_name: Where to save the file.
    """
    vocab = vocab_to_list(vocab) if isinstance(vocab, dict) else vocab
    wf.write(f"{len(vocab)} {vectors.shape[1]}\n".encode("utf-8"))
    for word, vector in zip(vocab, vectors):
        wf.write((word + " ").encode("utf-8") + vector.tobytes())
