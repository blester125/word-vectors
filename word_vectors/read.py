import re
import os
import mmap
import struct
import logging
import pathlib
from enum import Enum
from typing import Tuple, Union, TextIO, BinaryIO
import numpy as np
from file_or_name import file_or_name
from word_vectors import INT_SIZE, FLOAT_SIZE, DENSE_HEADER, Vocab, Vectors
from word_vectors.utils import _find_space, _find_max
from word_vectors.write import write_dense

FileType = Enum("FileType", "GLOVE W2V DENSE")

glove = re.compile(br"^[^ ]+? (-?\d+?\.\d+? )+", re.MULTILINE)
w2v = re.compile(br"^\d+? \d+?$", re.MULTILINE)
LOGGER = logging.getLogger("word_vectors")


@file_or_name(f="rb")
def sniff(f: Union[str, TextIO], buf_size: int = 1024) -> FileType:
    """Figure out what kind of vector file it is."""
    b = f.read(buf_size)
    if f.mode == "r":
        return FileType.GLOVE
    if w2v.match(b):
        return FileType.W2V
    elif glove.match(b):
        return FileType.GLOVE
    return FileType.DENSE


def read(f: Union[str, TextIO, BinaryIO], convert: bool = False, replace: bool = False) -> Tuple[Vocab, Vectors]:
    """Read vectors from file.

    Args:
        f: The file to read from.
        convert: Convert the vectors into the Dense format.
        replace: Replace the vector file with the Dense version.

    Returns:
        The vocab and vectors.
    """
    type_ = sniff(f)
    LOGGER.info("Sniffed word vector as type %s", type_)
    if type_ is FileType.GLOVE:
        reader = read_glove
    elif type_ is FileType.W2V:
        reader = read_w2v
    elif type_ is FileType.DENSE:
        reader = read_dense
        convert = False
        replace = False
    w, wv = reader(f)
    if convert:
        if isinstance(f, (str, pathlib.PurePath)):
            output = str(f)
        else:
            output = f.name
        if not replace:
            output = output + ".dense"
        len_ = _find_max(w.keys())
        write_dense(output, w, wv, len_)
    return w, wv


@file_or_name(f="r")
def read_glove(f: Union[str, TextIO]) -> Tuple[Vocab, Vectors]:
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


@file_or_name(f="rb")
def read_w2v(f: Union[str, BinaryIO]) -> Tuple[Vocab, Vectors]:
    """Read vectors from a word2vec file.

    Args:
        file_name: The file to read from

    Returns:
        The vocab and vectors.
    """
    words = {}
    vectors = []
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        header = m.readline()
        offset = m.tell()
        vocab, dim = map(int, header.decode("utf-8").split())
        size = FLOAT_SIZE * dim
        for _ in range(vocab):
            word, offset = _find_space(m, offset)
            if word not in words:
                words[word] = len(words)
                raw = m[offset : offset + size]
                vector = np.frombuffer(raw, dtype=np.float32)
                vectors.append(vector)
            offset = offset + size
    return words, np.vstack(vectors)


@file_or_name(f="rb")
def read_dense(f: Union[str, BinaryIO]) -> Tuple[Vocab, Vectors]:
    """Read vectors from a dense file.

    Args:
        file_name: The file to read from

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
            if word not in words:
                vector = np.frombuffer(line[length:], dtype=np.float32)
                words[word] = len(words)
                vectors.append(vector)
    return words, np.vstack(vectors)
