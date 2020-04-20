import re
import os
import mmap
import struct
import pathlib
from enum import Enum
from typing import Iterable, Tuple, Union, TextIO
import numpy as np
from file_or_name import file_or_name
from word_vectors import INT_SIZE, FLOAT_SIZE, DENSE_HEADER, Vocab, Vectors
from word_vectors.write import write_dense

FileType = Enum("FileType", "GLOVE W2V DENSE")

glove = re.compile(br"^[^ ]+? (-?\d+?\.\d+? )+", re.MULTILINE)
w2v = re.compile(br"^\d+? \d+?$", re.MULTILINE)


@file_or_name(f="rb")
def sniff(f: Union[str, TextIO]) -> FileType:
    """Figure out what kind of vector file it is."""
    b = f.read(1024)
    if f.mode == "r":
        return FileType.GLOVE
    if w2v.match(b):
        return FileType.W2V
    elif glove.match(b):
        return FileType.GLOVE
    return FileType.DENSE


def find_max(words: Iterable[str]) -> int:
    """Get the max length of words (as bytes)."""
    return max(map(len, map(lambda x: x.encode("utf-8"), words)))


def read(f: Union[str, TextIO], convert: bool = False, replace: bool = False) -> Tuple[Vocab, Vectors]:
    """Read vectors from file.

    Args:
        f: The file to read from.
        convert: Convert the vectors into the Dense format.
        replace: Replace the vector file with the Dense version.

    Returns:
        The vocab and vectors.
    """
    w = None
    wv = None
    len_ = None
    type_ = sniff(f)
    if type_ is FileType.GLOVE:
        w, wv = read_glove(f)
    elif type_ is FileType.W2V:
        w, wv, len_ = read_w2v(f, stats=True)
    elif type_ is FileType.DENSE:
        w, wv, len_ = read_dense(f, stats=True)
        convert = False
        replace = False
    if convert:
        if isinstance(f, (str, pathlib.PurePath)):
            output = str(f)
        else:
            output = f.name
        if not replace:
            output = output + ".dense"
        if len_ is None:
            len_ = find_max(w)
        write_dense(output, w, wv, len_)
    return w, wv


@file_or_name(f="r")
def read_glove(f: str) -> Tuple[Vocab, Vectors]:
    """Read vectors from a glove file.

    Args:
        file_name: The file to read from

    Returns:
        The vocab and vectors.
    """
    words = {}
    vectors = []
    i = 0
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        for line in iter(m.readline, b""):
            line = line.decode("utf-8")
            line = line.rstrip("\n")
            word, *vector = line.split(" ")
            if word not in words:
                words[word] = i
                vectors.append(np.asarray(vector, dtype=np.float32))
                i += 1
    return words, np.vstack(vectors)


def _find_space(m, offset) -> Tuple[str, int, int]:
    i = offset + 1
    while m[i : i + 1] != b" ":
        i += 1
    word = m[offset:i].decode("utf-8")
    return word, i - offset, i + 1


@file_or_name(f="rb")
def read_w2v(f: str, stats: bool = False) -> Tuple[Vocab, Vectors]:
    """Read vectors from a word2vec file.

    Args:
        file_name: The file to read from
        stats: Should you return the length of the longest word.

    Returns:
        The vocab and vectors.
    """
    words = {}
    vectors = []
    max_len = 0
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        header = m.readline()
        offset = m.tell()
        vocab, dim = map(int, header.decode("utf-8").split())
        size = FLOAT_SIZE * dim
        for i in range(vocab):
            word, word_len, offset = _find_space(m, offset)
            if word_len > max_len:
                max_len = word_len
            raw = m[offset : offset + size]
            vector = np.frombuffer(raw, dtype=np.float32)
            words[word] = i
            vectors.append(vector)
            offset = offset + size
    if stats:
        return words, np.vstack(vectors), max_len
    return words, np.vstack(vectors)


@file_or_name(f="rb")
def read_dense(f: str, stats: bool = False) -> Tuple[Vocab, Vectors]:
    """Read vectors from a dense file.

    Args:
        file_name: The file to read from
        stats: Should you return the length of the max word?

    Note:
        The dense file format is pretty simple. The first three bytes are
        the vocab size, vector size, and max word length as little endian
        values. Each "line" is a word in max word bytes followed by the
        vector of size vector size * 4 (size of np.float32)

    Returns:
        The vocab and vectors.
    """
    words = {}
    vectors = []
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        offset = INT_SIZE * DENSE_HEADER
        vocab, dim, length = struct.unpack("<iii", m[:offset])
        size = FLOAT_SIZE * dim
        for i in range(vocab):
            start = offset + i * (length + size)
            line = m[start : start + length + size]
            word = line[:length].decode("utf-8").strip(" ")
            vector = np.frombuffer(line[length:], dtype=np.float32)
            words[word] = i
            vectors.append(vector)
    if stats:
        return words, np.vstack(vectors), length
    return words, np.vstack(vectors)
