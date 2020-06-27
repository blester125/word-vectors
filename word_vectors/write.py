"""Write Word Vectors to a file.

We provide the main :py:func:`~word_vectors.write.write` function that can write
to various vector serialization formats based on the passed :py:attr:`~word_vecotrs.FileType`.
There are also several convenience functions for writing specific formats.
"""

import struct
from operator import itemgetter
from typing import Union, IO, TextIO, BinaryIO, Optional, Iterable
from file_or_name import file_or_name
from word_vectors import Vocab, Vectors, FileType, LEADER_MAGIC_NUMBER
from word_vectors.utils import to_vocab


def write(
    wf: Union[str, IO],
    vocab: Union[Vocab, Iterable[str]],
    vectors: Vectors,
    file_type: FileType,
    max_len: Optional[int] = None,
):
    """Write word vectors to a file.

    This function dispatches to on of the following word vector format writers based
    on the file of ``file_type``.

    - :py:func:`~word_vectors.write.write_glove`
    - :py:func:`~word_vectors.write.write_w2v_text`
    - :py:func:`~word_vectors.write.write_w2v`
    - :py:func:`~word_vectors.write.write_leader`

    Args:
        wf: The file we are writing to.
        vocab: The vocab mapping words -> ints.
        vectors: The vectors as a ``np.ndarray``.
        file_type: The format to use when writing the vectors to disk.
        max_len: The maximum length of a word in vocab. Only used when writing Leader vectors.

    Raises:
        ValueError: If the an unsupported file type is passed
    """
    if file_type is FileType.GLOVE:
        write_glove(wf, vocab, vectors)
    elif file_type is FileType.W2V:
        write_w2v(wf, vocab, vectors)
    elif file_type is FileType.W2V_TEXT or file_type is FileType.FASTTEXT or file_type is FileType.NUMBERBATCH:
        write_w2v_text(wf, vocab, vectors)
    elif file_type is FileType.LEADER:
        write_leader(wf, vocab, vectors)
    else:
        raise ValueError(f"FileType not understood, got: {file_type}")


@file_or_name(wf="w")
def write_glove(wf: Union[str, TextIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors):
    """Write vectors to a glove file.

    See :py:func:`word_vectors.read.read_glove` for a description of the file format and
    examples of common pre-trained embeddings that use this format.

    Args:
        wf: The file we are writing to
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
    """
    vocab = to_vocab(vocab) if not isinstance(vocab, dict) else vocab
    for word, idx in sorted(vocab.items(), key=itemgetter(1)):
        wf.write(" ".join([word, *map(str, vectors[idx])]) + "\n")


@file_or_name(wf="w")
def write_w2v_text(wf: Union[str, TextIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors):
    """Write vectors in the word2vec format in a text file.

    See :py:func:`word_vectors.read.read_w2v_text` for a description of the file format and
    examples of common pre-trained embeddings that use this format.

    Args:
        wf: The file we are writing to
        vocab: The vocab of words -> ints
        vectors: The vectors we are writing
    """
    wf.write(f"{len(vocab)} {vectors.shape[1]}\n")
    write_glove(wf, vocab, vectors)


@file_or_name(wf="wb")
def write_w2v(wf: Union[str, BinaryIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors):
    """Write vectors to the word2vec format as a binary file.

    See :py:func:`word_vectors.read.read_w2v` for a description of the file format and
    examples of common pre-trained embeddings that use this format.

    Args:
        wf: The file we are writing to
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
    """
    vocab = to_vocab(vocab) if not isinstance(vocab, dict) else vocab
    wf.write(f"{len(vocab)} {vectors.shape[1]}\n".encode("utf-8"))
    for word, idx in sorted(vocab.items(), key=itemgetter(1)):
        wf.write((word + " ").encode("utf-8") + vectors[idx].tobytes())


@file_or_name(wf="wb")
def write_leader(wf: Union[str, BinaryIO], vocab: Union[Vocab, Iterable[str]], vectors: Vectors):
    """Write vectors to a leader file.

    See :py:func:`word_vectors.read.read_leader` for a description of the file format.

    Args:
        wf: The file we are writing to.
        vocab: The vocab of words -> ints.
        vectors: The vectors as a np.ndarray.
        max_len: The longest length of the words as (``utf-8``) bytes.
    """
    vocab = to_vocab(vocab) if not isinstance(vocab, dict) else vocab
    for val in (LEADER_MAGIC_NUMBER, len(vocab), vectors.shape[1]):
        wf.write(struct.pack("<Q", val))
    for word, idx in sorted(vocab.items(), key=itemgetter(1)):
        word = word.encode("utf-8")
        word_len = len(word)
        wf.write(struct.pack("<I", word_len))
        wf.write(word)
        wf.write(vectors[idx].tobytes())
