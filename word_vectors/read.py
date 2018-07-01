import re
import mmap
import struct
from enum import Enum
from typing import Iterable, Tuple
import numpy as np
from word_vectors import INT_SIZE, Vocab, Vects
from word_vectors.write import write_dense

Vectors = Enum("Vectors", "GLOVE W2V DENSE")

glove = re.compile(br"^\w+? (-?\d+?\.\d+? )+", re.MULTILINE)
w2v = re.compile(br"^\d+? \d+?$", re.MULTILINE)

def read(
        file_name: str,
        convert: bool=True, replace: bool=True
) -> Tuple[Vocab, Vects]:
    """Read vectors from file.

    Args:
        file_name: The file to read from.
        convert: Convert the vectors into the Dense format.
        replace: Replace the vector file with the Dense version.

    Returns:
        The vocab and vectors.
    """
    w = None
    wv = None
    len_ = None
    type_ = sniff(file_name)
    if type_ is Vectors.GLOVE:
        w, wv = read_glove(file_name)
    elif type_ is Vectors.W2V:
        w, wv, len_ = read_w2v(file_name, stats=True)
    elif type_ is Vectors.DENSE:
        w, wv, len_ = read_dense(file_name, stats=True)
        convert = False
        replace = False
    if convert:
        if not replace:
            file_name = file_name + '.dense'
        if len_ is None:
            len_ = find_max(w)
        write_dense(w, wv, len_, file_name)
    return w, wv

def read_glove(file_name: str) -> Tuple[Vocab, Vects]:
    """Read vectors from a glove file.

    Args:
        file_name: The file to read from

    Returns:
        The vocab and vectors.
    """
    words = {}
    vectors = []
    i = 0
    with open(file_name, 'r') as f:
        for line in f:
            line = line.rstrip("\n")
            word, *vector = line.split(" ")
            if word not in words:
                words[word] = i
                vectors.append(np.asarray(vector, dtype=np.float32))
                i += 1
    return words, np.vstack(vectors)

def _find_space(f) -> Tuple[str, int]:
    """Read forward in a file until finding a space."""
    word = bytearray()
    char = f.read(1)
    while char != b' ':
        word.extend(char)
        char = f.read(1)
    length = len(word)
    word = word.decode('utf-8')
    return word.strip(' \n'), length

def read_w2v(file_name: str, stats: bool=False) -> Tuple[Vocab, Vects]:
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
    with open(file_name, 'rb') as f:
        header = f.readline()
        vocab, dim = map(int, header.decode('utf-8').split())
        size = INT_SIZE * dim
        for i in range(vocab):
            word, word_len = _find_space(f)
            if word_len > max_len:
                max_len = word_len
            raw = f.read(size)
            vector = np.fromstring(raw, dtype=np.float32)
            words[word] = i
            vectors.append(vector)
    if stats:
        return words, np.vstack(vectors), max_len
    return words, np.vstack(vectors)

def sniff(file_name: str) -> Vectors:
    """Figure out what kind of vector file it is."""
    b = open(file_name, 'rb').read(1024)
    if w2v.match(b):
        return Vectors.W2V
    elif glove.match(b):
        return Vectors.GLOVE
    return Vectors.DENSE

def find_max(words: Iterable[str]) -> int:
    """Get the max length of words (as bytes)."""
    return max(map(len, map(lambda x: x.encode('utf-8'), words)))

def read_dense(file_name: str, stats: bool=False) -> Tuple[Vocab, Vects]:
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
    with open(file_name, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
            offset = INT_SIZE * 3
            vocab, dim, length = struct.unpack('<iii', m[:offset])
            size = INT_SIZE * dim
            for i in range(vocab):
                start = offset + i * (length + size)
                line = m[start:start+length+size]
                word = line[:length].decode('utf-8').strip(' ')
                vector = np.fromstring(line[length:], dtype=np.float32)
                words[word] = i
                vectors.append(vector)
    if stats:
        return words, np.vstack(vectors), length
    return words, np.vstack(vectors)
