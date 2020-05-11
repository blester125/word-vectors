"""Write Word Vectors to a file."""

import struct
from operator import itemgetter
from typing import Union, IO, TextIO, BinaryIO, Optional, Iterable
from file_or_name import file_or_name
from word_vectors import Vocab, Vectors, FileType
from word_vectors.utils import find_max


def _pad(word: str, max_len: int) -> bytes:
    """Pad a word out so the byte representation is max_len.

    Args:
        word: The word we are padding out and converting to binary
        max_len: How far to pad the string

    Returns:
        The word as bytes and extended.
    """
    byte_words = word.encode("utf-8")
    return byte_words + b" " * (max_len - len(byte_words))


def to_vocab(words: Iterable[str]) -> Vocab:
    """Convert a series of words to a vocab mapping strings to ints.

    Args:
        words: The words in the vocab

    Returns:
        The Vocabulary
    """
    return {w: i for i, w in enumerate(words)}


def write(
    wf: Union[str, IO],
    vocab: Union[Vocab, Iterable[str]],
    vectors: Vectors,
    file_type: FileType,
    max_len: Optional[int] = None,
) -> None:
    if file_type is FileType.GLOVE:
        write_glove(wf, vocab, vectors)
    elif file_type is FileType.W2V:
        write_w2v(wf, vocab, vectors)
    elif file_type is FileType.W2V_TEXT:
        write_w2v_text(wf, vocab, vectors)
    elif file_type is FileType.DENSE:
        write_dense(wf, vocab, vectors, max_len)
    else:
        raise ValueError(f"FileType not understood, got: {file_type}")


@file_or_name(wf="wb")
def write_dense(
    wf: Union[str, BinaryIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors, max_len: Optional[int] = None
) -> None:
    """Write vectors to a dense file.

    Args:
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        max_len: The longest length of the words as bytes.
        file_name: Where to save the file.
    """
    vocab = to_vocab(vocab) if not isinstance(vocab, dict) else vocab
    max_len = find_max(vocab) if max_len is None else max_len
    for val in (len(vocab), vectors.shape[1], max_len):
        wf.write(struct.pack("<i", val))
    for word, idx in sorted(vocab.items(), key=itemgetter(1)):
        word = _pad(word, max_len)
        wf.write(word)
        wf.write(vectors[idx].tobytes())


@file_or_name(wf="w")
def write_glove(wf: Union[str, TextIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors) -> None:
    """Write vectors to a glove file.

    Args:
        wf: The file we are writing to
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
    """
    vocab = to_vocab(vocab) if not isinstance(vocab, dict) else vocab
    for word, idx in sorted(vocab.items(), key=itemgetter(1)):
        wf.write(" ".join([word, *map(str, vectors[idx])]) + "\n")


@file_or_name(wf="wb")
def write_w2v(wf: Union[str, BinaryIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors) -> None:
    """Write vectors to the word2vec format as a binary file.

    Args:
        wf: The file we are writing to
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
    """
    vocab = to_vocab(vocab) if not isinstance(vocab, dict) else vocab
    wf.write(f"{len(vocab)} {vectors.shape[1]}\n".encode("utf-8"))
    for word, idx in sorted(vocab.items(), key=itemgetter(1)):
        wf.write((word + " ").encode("utf-8") + vectors[idx].tobytes())


@file_or_name(wf="w")
def write_w2v_text(wf: Union[str, TextIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors) -> None:
    """Write vectors in the word2vec format in a text file.

    Args:
        wf: The file we are writing to
        vocab: The vocab of words -> ints
        vectors: The vectors we are writing
    """
    wf.write(f"{len(vocab)} {vectors.shape[1]}\n")
    write_glove(wf, vocab, vectors)
