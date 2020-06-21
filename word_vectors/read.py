"""Read word vectors from a file.

We provide a main :py:func:`~word_vectors.read.read` function for reading vectors
from a file. The serialization format can be explicitly provided with by passing a
:py:attr:`~word_vectors.FileType` or automatically inferred using
:py:func:`~word_vectors.read.sniff`. There are also several provided convenience
functions for reading from specific formats.
"""

import re
import os
import mmap
import struct
import logging
import pathlib
from typing import Tuple, Union, IO, TextIO, BinaryIO, Optional, Iterator, Callable
import numpy as np
from file_or_name import file_or_name
from word_vectors import LONG_SIZE, FLOAT_SIZE, DENSE_HEADER, Vocab, Vectors, FileType, DENSE_MAGIC_NUMBER
from word_vectors.utils import find_space, is_binary, bookmark, uniform_initializer


GLOVE_TEXT = re.compile(r"^[^ ]+? (-?\d+?\.\d+? )+", re.MULTILINE)
GLOVE_BIN = re.compile(br"^[^ ]+? (-?\d+?\.\d+? )+", re.MULTILINE)
W2V_TEXT = re.compile(r"^\d+ \d+$", re.MULTILINE)
W2V_BIN = re.compile(br"^\d+ \d+$", re.MULTILINE)

LOGGER = logging.getLogger("word_vectors")


# We don't know what mode to open the file in (text for things like Glove while
# binary for things like Word2Vec or Dense) we can't use the `@file_or_name`
# decorator directly but all the functions we call use that so we can handle
# all the file formats.
def read(f: Union[str, TextIO, BinaryIO], file_type: Optional[FileType] = None) -> Tuple[Vocab, Vectors]:
    """Read vectors from a file.

    This function can dispatch to one of the following word vector format readers:

    - :py:func:`~word_vectors.read.read_glove`
    - :py:func:`~word_vectors.read.read_w2v_text`
    - :py:func:`~word_vectors.read.read_w2v`
    - :py:func:`~word_vectors.read.read_dense`

    Check the documentation of a specific reader to see a description of the file
    format as well as common pre-trained vectors that ship with this format.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Note:
        Without a specified file type this function uses :py:func:`word_vectors.read.sniff`
        to determine the word vector format and dispatches to the appropriate reader.

        I haven't seen a sniffing failure but if your file type can't be determined you
        can pass the ``file_type`` explicitly or call the specific reading function yourself.

    Args:
        f: The file to read from.
        file_type: The vector file format. If ``None`` the file is sniffed to determine
            format.

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    if file_type is None:
        file_type = sniff(f)
        LOGGER.info("Sniffed word vector as type %s", file_type)
    if file_type is FileType.GLOVE:
        line_reader = read_glove_lines
    elif file_type is FileType.W2V_TEXT or file_type is FileType.FASTTEXT or file_type is FileType.NUMBERBATCH:
        line_reader = read_w2v_text_lines
    elif file_type is FileType.W2V:
        line_reader = read_w2v_lines
    elif file_type is FileType.DENSE:
        line_reader = read_dense_lines
    else:
        raise ValueError(f"Unknown vector format, got: {line_reader}")
    return _read(f, line_reader)


def read_with_vocab(
    f: Union[str, IO],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray] = uniform_initializer(0.25),
    keep_extra: bool = False,
    file_type: Optional[FileType] = None,
) -> Tuple[Vocab, Vectors]:
    """Read vectors from a file subject user provided vocabulary constraints.

    This function can dispatch to one of the following word vector format readers:

    - :py:func:`~word_vectors.read.read_glove_with_vocab`
    - :py:func:`~word_vectors.read.read_w2v_text_with_vocab`
    - :py:func:`~word_vectors.read.read_w2v_with_vocab`
    - :py:func:`~word_vectors.read.read_dense_with_vocab`

    Check the documentation of a specific reader to see a description of the file
    format as well as common pre-trained vectors that ship with this format.

    When provided a vocabulary this function will not reorder it. If you pass in that
    the word ``dog`` is index ``12`` then in the resulting vocabulary it will still
    be index ``12``.

    When collecting extra vocabulary (words that are in the pre-trained embeddings but
    not in the user vocab) these will all be at the end of the vocabulary. Again the
    indices of user provided words will not change.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Note:
        Without a specified file type this function uses :py:func:`word_vectors.read.sniff`
        to determine the word vector format and dispatches to the appropriate reader.

        I haven't seen a sniffing failure but if your file type can't be determined you
        can pass the ``file_type`` explicitly or call the specific reading function yourself.

    Args:
        f: The file to read from.
        user_vocab: A specific vocabulary the user wants to extract form the pre-trained
            embeddings.
        initializer: A function that takes the vector size and generates a new vector.
            this is used to generate a representation for a word in the user vocab that
            is not in the pre-train embeddings.
        keep_extra: Should you also include vectors that are in the pre-trained embedding
            but not in the user provided vocab?
        file_type: The vector file format. If ``None`` the file is sniffed to determine
            format.

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    if file_type is None:
        file_type = sniff(f)
        LOGGER.info("Sniffed word vector as type %s", file_type)
    if file_type is FileType.GLOVE:
        line_reader = read_glove_lines
    elif file_type is FileType.W2V_TEXT or file_type is FileType.FASTTEXT or file_type is FileType.NUMBERBATCH:
        line_reader = read_w2v_text_lines
    elif file_type is FileType.W2V:
        line_reader = read_w2v_lines
    elif file_type is FileType.DENSE:
        line_reader = read_dense_lines
    else:
        raise ValueError(f"Unknown vector format, got: {line_reader}")
    if keep_extra:
        return _read_with_vocab_extra(f, line_reader, user_vocab, initializer)
    return _read_with_vocab(f, line_reader, user_vocab, initializer)


def _read(
    f: Union[str, IO], line_reader: Callable[[Union[str, IO]], Iterator[Tuple[str, np.ndarray]]]
) -> Tuple[Vocab, Vectors]:
    words = {}
    vectors = []
    for word, vector in line_reader(f):
        if word not in words:
            words[word] = len(words)
            vectors.append(vector)
    return words, np.vstack(vectors)


def _read_with_vocab(
    f: Union[str, IO],
    line_reader: Callable[[Union[str, IO]], Iterator[Tuple[str, np.ndarray]]],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray],
) -> Tuple[Vocab, Vectors]:
    vectors = [None for _ in range(len(user_vocab))]
    for word, vector in line_reader(f):
        vector_size = len(vector)
        if word in user_vocab:
            idx = user_vocab[word]
            if vectors[idx] is None:
                vectors[idx] = vector
    for i, vector in enumerate(vectors):
        if vector is None:
            vectors[i] = initializer(vector_size)
    return user_vocab, np.vstack(vectors)


def _read_with_vocab_extra(
    f: Union[str, IO],
    line_reader: Callable[[Union[str, IO]], Iterator[Tuple[str, np.ndarray]]],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray],
) -> Tuple[Vocab, Vectors]:
    user_vocab_size = len(user_vocab)
    vectors = [None for _ in range(len(user_vocab))]
    extra_words = {}
    extra_vectors = []
    for word, vector in line_reader(f):
        vector_size = len(vector)
        if word in user_vocab:
            idx = user_vocab[word]
            if vectors[idx] is None:
                vectors[idx] = vector
        else:
            if word not in extra_words:
                extra_words[word] = len(extra_words)
                extra_vectors.append(vector)
    for i, vector in enumerate(vectors):
        if vector is None:
            vectors[i] = initializer(vector_size)
    for word, idx in extra_words.items():
        user_vocab[word] = user_vocab_size + idx
    vectors.extend(extra_vectors)
    return user_vocab, np.vstack(vectors)


@file_or_name
def read_glove_lines(f: Union[str, TextIO]) -> Iterator[Tuple[str, np.ndarray]]:
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        for line in iter(m.readline, b""):
            line = line.decode("utf-8")
            line = line.rstrip("\n")
            word, *vector = line.split(" ")
            yield word, np.asarray(vector, dtype=np.float32)


@file_or_name
def read_w2v_text_lines(f: Union[str, TextIO]) -> Iterator[Tuple[str, np.ndarray]]:
    # Because the ``mmap`` call starts at the beginning of the file and the offset
    # needs to be a multiple of ``ALLOCATIONGRANULARITY`` we can't start from the offset
    # that an ``f.readline()`` would give us. This means we can't just advance by one line
    # and then call read_glove so I had to duplicate code here :/
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        _ = m.readline()
        for line in iter(m.readline, b""):
            line = line.decode("utf-8")
            line = line.rstrip("\n")
            word, *vector = line.split(" ")
            yield word, np.asarray(vector, dtype=np.float32)


@file_or_name(f="rb")
def read_w2v_lines(f: Union[str, BinaryIO]) -> Iterator[Tuple[str, np.ndarray]]:
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        header = m.readline()
        offset = m.tell()
        vocab, dim = map(int, header.decode("utf-8").split())
        size = FLOAT_SIZE * dim
        for _ in range(vocab):
            word, offset = find_space(m, offset)
            raw = m[offset : offset + size]
            vector = np.frombuffer(raw, dtype=np.float32)
            yield word, vector
            offset = offset + size


@file_or_name(f="rb")
def read_dense_lines(f: Union[str, BinaryIO]) -> Iterator[Tuple[str, np.ndarray]]:
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        offset = LONG_SIZE * DENSE_HEADER
        vocab, dim, length = read_dense_header(m[:offset])
        size = FLOAT_SIZE * dim
        for i in range(vocab):
            start = offset + i * (length + size)
            line = m[start : start + length + size]
            word = line[:length].rstrip(b" ").decode("utf-8")
            vector = np.frombuffer(line[length:], dtype=np.float32)
            yield word, vector


@file_or_name
def read_glove(f: Union[str, TextIO]) -> Tuple[Vocab, Vectors]:
    """Read vectors from a glove file.

    The GloVe format is a pure text format. Each (word, vector) pair is represented
    by a single line in the file. The line starts with the word, a space, and then
    the float32 text representations of the elements in the vector associated with
    that word. Each of these vector elements are also separated with a space.

    The main vectors distributed in this format are the `GloVe`_ vectors
    `(Pennington, et. al., 2014)`_

    .. _GloVe: https://nlp.stanford.edu/projects/glove/
    .. _(Pennington, et. al., 2014): https://www.aclweb.org/anthology/D14-1162/

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    return _read(f, read_glove_lines)


@file_or_name
def read_glove_with_vocab(
    f: Union[str, TextIO],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray] = uniform_initializer(0.25),
    keep_extra: bool = False,
) -> Tuple[Vocab, Vectors]:
    """Read vectors from a glove file subject to user vocabulary constraints.

    See :py:func:`~word_vectors.read.read_glove` for a description of the file format
    and common pre-train embeddings that use this format.

    When provided a vocabulary this function will not reorder it. If you pass in that
    the word ``dog`` is index ``12`` then in the resulting vocabulary it will still
    be index ``12``.

    When collecting extra vocabulary (words that are in the pre-trained embeddings but
    not in the user vocab) these will all be at the end of the vocabulary. Again the
    indices of user provided words will not change.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from.
        user_vocab: A specific vocabulary the user wants to extract form the pre-trained
            embeddings.
        initializer: A function that takes the vector size and generates a new vector.
            this is used to generate a representation for a word in the user vocab that
            is not in the pre-train embeddings.
        keep_extra: Should you also include vectors that are in the pre-trained embedding
            but not in the user provided vocab?

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    if keep_extra:
        return _read_with_vocab_extra(f, read_glove_lines, user_vocab, initializer)
    return _read_with_vocab(f, read_glove_lines, user_vocab, initializer)


@file_or_name
def read_w2v_text(f: Union[str, TextIO]) -> Tuple[Vocab, Vectors]:
    """Read vectors from a text based w2v file.

    One of two different vector serialization formats introduced in the
    `word2vec software`_ `(Mikolov, et. al., 2013)`_.

    The word2vec text format is a pure text format. The first line is two
    integers, represented as text and separated by a space, that specify the
    number of types in the vocabulary and the size of the word vectors
    respectively. Each following line represents a (word, vector) pair. The
    line stars with the word, a space, and then the float 32 text representations
    of the elements in the vector associated with that word. Each of these vector
    elements are also separated with a space.

    One can see that that this is actually the same as the
    :py:func:`GloVe <word_vectors.read.read_glove>` format except that in GloVe
    they removed the header line.

    The main embeddings distributed in this format are
    `FastText`_ `(Bojanowski, et. al., 2017)`_ and
    `NumberBatch`_ `(Speer, et. al., 2017)`_


    .. _word2vec software: https://code.google.com/archive/p/word2vec/
    .. _(Mikolov, et. al., 2013): https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality
    .. _FastText: https://fasttext.cc/
    .. _(Bojanowski, et. al., 2017): https://www.aclweb.org/anthology/Q17-1010/
    .. _NumberBatch: https://github.com/commonsense/conceptnet-numberbatch
    .. _(Speer, et. al., 2017): https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    return _read(f, read_w2v_text_lines)


@file_or_name
def read_w2v_text_with_vocab(
    f: Union[str, TextIO],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray] = uniform_initializer(0.25),
    keep_extra: bool = False,
) -> Tuple[Vocab, Vectors]:
    """Read vectors from a Word2Vec text file subject to user vocabulary constraints.

    See :py:func:`~word_vectors.read.read_w2v_text` for a description of the file format
    and common pre-train embeddings that use this format.

    When provided a vocabulary this function will not reorder it. If you pass in that
    the word ``dog`` is index ``12`` then in the resulting vocabulary it will still
    be index ``12``.

    When collecting extra vocabulary (words that are in the pre-trained embeddings but
    not in the user vocab) these will all be at the end of the vocabulary. Again the
    indices of user provided words will not change.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from.
        user_vocab: A specific vocabulary the user wants to extract form the pre-trained
            embeddings.
        initializer: A function that takes the vector size and generates a new vector.
            this is used to generate a representation for a word in the user vocab that
            is not in the pre-train embeddings.
        keep_extra: Should you also include vectors that are in the pre-trained embedding
            but not in the user provided vocab?

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    if keep_extra:
        return _read_with_vocab_extra(f, read_w2v_text_lines, user_vocab, initializer)
    return _read_with_vocab(f, read_w2v_text_lines, user_vocab, initializer)


@file_or_name(f="rb")
def read_w2v(f: Union[str, BinaryIO]) -> Tuple[Vocab, Vectors]:
    """Read vectors from a word2vec file.

    One of two different vector serialization formats introduced in the
    `word2vec software`_ `(Mikolov, et. al., 2013)`_.

    The word2vec binary format is a mix of textual an binary representations.
    The first line is two integers (as text, separated be a space) representing
    the number of types in the vocabulary and the size of the word vectors
    respectively. (word, vector) pairs follow. The word is represented as text
    and a space. After the space each element of a vector is represented as a
    binary float32.

    The most well-known pre-trained embeddings distributed in this format are
    the `GoogleNews`_ vectors.

    Note:

        There is no formal definition of this file format, the only definitive
        reference on it is the original implementation in the `word2vec software`_

        Due to the lack of a definition (an no special handling of it in the code)
        there is no explicit statements about the endianness of the binary representations.
        Most code just uses the ``numpy.from_buffer`` and that seems to work now that
        most people have little-endian machines. However due to the lack of explicit
        direction on this encoding I would advise caution when loading vectors that
        were trained on big-endian hardware.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    .. _word2vec software: https://code.google.com/archive/p/word2vec/
    .. _(Mikolov, et. al., 2013): https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality
    .. _GoogleNews: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

    Args:
        f: The file to read from

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    return _read(f, read_w2v_lines)


@file_or_name(f="rb")
def read_w2v_with_vocab(
    f: Union[str, BinaryIO],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray] = uniform_initializer(0.25),
    keep_extra: bool = False,
) -> Tuple[Vocab, Vectors]:
    """Read vectors from a Word2Vec file subject to user vocabulary constraints.

    See :py:func:`~word_vectors.read.read_w2v` for a description of the file format
    and common pre-train embeddings that use this format.

    When provided a vocabulary this function will not reorder it. If you pass in that
    the word ``dog`` is index ``12`` then in the resulting vocabulary it will still
    be index ``12``.

    When collecting extra vocabulary (words that are in the pre-trained embeddings but
    not in the user vocab) these will all be at the end of the vocabulary. Again the
    indices of user provided words will not change.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from.
        user_vocab: A specific vocabulary the user wants to extract form the pre-trained
            embeddings.
        initializer: A function that takes the vector size and generates a new vector.
            this is used to generate a representation for a word in the user vocab that
            is not in the pre-train embeddings.
        keep_extra: Should you also include vectors that are in the pre-trained embedding
            but not in the user provided vocab?

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    if keep_extra:
        return _read_with_vocab_extra(f, read_w2v_lines, user_vocab, initializer)
    return _read_with_vocab(f, read_w2v_lines, user_vocab, initializer)


@file_or_name(f="rb")
def read_dense(f: Union[str, BinaryIO]) -> Tuple[Vocab, Vectors]:
    """Read vectors from a dense file.

    This is our fully binary vector format.

    The first line is a header for the dense format and it is a
    4-tuple. The elements of
    this tuple are: A magic number, the size of the vocabulary, the
    size of the vectors, and the length of the longest word in the
    vocabulary (this length when represented as ``utf-8`` bytes
    rather than as Unicode codepoints). These numbers are represented
    as little-endian unsigned long longs that have a size of 8 bytes.

    Following the header the are (word, vector) pairs. The words are
    stored as ``utf-8`` bytes. The trick is that they are padded out to
    be a consistent length (this length is the length of the longest
    word in the vocabulary). After the word the vector is stored where
    each element is a little-endian float32 (4 bytes).

    The consistent word lengths lets us calculate the offset to any
    word quickly without having to iterate over the characters to
    find the space as in the word2vec binary format. Finding the
    word at index ``i`` can be done with some offset math.
    ``offset for i = header length + i * (max length + vector size)``

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from

    Returns:
        The vocab and vectors.
    """
    return _read(f, read_dense_lines)


@file_or_name(f="rb")
def read_dense_with_vocab(
    f: Union[str, BinaryIO],
    user_vocab: Vocab,
    initializer: Callable[[int], np.ndarray] = uniform_initializer(0.25),
    keep_extra: bool = False,
) -> Tuple[Vocab, Vectors]:
    """Read vectors from a Dense file subject to user vocabulary constraints.

    See :py:func:`~word_vectors.read.read_dense` for a description of the file format
    and common pre-train embeddings that use this format.

    When provided a vocabulary this function will not reorder it. If you pass in that
    the word ``dog`` is index ``12`` then in the resulting vocabulary it will still
    be index ``12``.

    When collecting extra vocabulary (words that are in the pre-trained embeddings but
    not in the user vocab) these will all be at the end of the vocabulary. Again the
    indices of user provided words will not change.

    Note:
       In the case of duplicated words in the saved vectors we use the index
       and associated vector from the first occurrence of the word.

    Args:
        f: The file to read from.
        user_vocab: A specific vocabulary the user wants to extract form the pre-trained
            embeddings.
        initializer: A function that takes the vector size and generates a new vector.
            this is used to generate a representation for a word in the user vocab that
            is not in the pre-train embeddings.
        keep_extra: Should you also include vectors that are in the pre-trained embedding
            but not in the user provided vocab?

    Returns:
        The vocab and vectors. The vocab is a mapping from word to integer and
        vectors are a numpy array of shape ``[vocab size, vector size]``. The
        vocab gives the index offset into the vector matrix for some word.
    """
    if keep_extra:
        return _read_with_vocab_extra(f, read_dense_lines, user_vocab, initializer)
    return _read_with_vocab(f, read_dense_lines, user_vocab, initializer)


@file_or_name(f="rb")
def sniff(f: Union[str, TextIO], buf_size: int = 1024) -> FileType:
    """Figure out what kind of vector file it is.

    Args:
        f: The file we are sniffing.
        buf_size: How many bytes to read in when sniffing the file.

    Returns:
        The guessed file type.
    """
    with bookmark(f):
        b = f.read(buf_size)
    # Because we support reading from an already open file we can't just start applying
    # the bytes based regexs to the file, if the file was not opened in binary mode use
    # the normal unicode string regexs
    if "b" not in f.mode:
        if GLOVE_TEXT.match(b):
            return FileType.GLOVE
        if W2V_TEXT.match(b):
            return FileType.W2V_TEXT
    else:
        if is_binary(f):
            if W2V_BIN.match(b):
                return FileType.W2V
            if verify_dense(b):
                return FileType.DENSE
        else:
            if W2V_BIN.match(b):
                return FileType.W2V_TEXT
            if GLOVE_BIN.match(b):
                return FileType.GLOVE
    raise ValueError(f"Could not determine file format for {f.name}")


def read_dense_header(buf: bytes) -> Tuple[int, int, int]:
    """Read the header from the dense file.

    The header for the dense format is a 4-tuple. The elements of
    this tuple are: A magic number, the size of the vocabulary, the
    size of the vectors, and the length of the longest word in the
    vocabulary (this length when represented as ``utf-8`` bytes
    rather than as Unicode codepoints). These numbers are represented
    as little-endian unsigned long longs that have a size of 8 bytes.

    Note:
        The magic number if used to make sure this is can actual
        file and not just trying to extract word vectors from a
        random binary file. The Magic Number is ``2283``.

    Args:
        buf: The beginning of the file we are reading the header from.

    Returns:
        The vocab size, the vector size, and the maximum length of any of the words

    Raises:
        ValueError: If the magic number doesn't match.
    """
    magic, vocab, dim, length = struct.unpack("<QQQQ", buf[: LONG_SIZE * DENSE_HEADER])
    if magic != DENSE_MAGIC_NUMBER:
        raise ValueError(f"Magic Number read does not match expected. Expected: `{DENSE_MAGIC_NUMBER}` Got: `{magic}`")
    return vocab, dim, length


def verify_dense(buf: bytes) -> bool:
    """Check if a file is in the dense format by comparing the magic number.

    Args:
        buf: The beginning of the file we are trying to determine if the it
        is a Dense formatted file.

    Returns:
        True if the magic number matched, False otherwise.
    """
    try:
        read_dense_header(buf)
        return True
    except ValueError:
        return False
