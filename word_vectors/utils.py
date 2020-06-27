"""Utilities for working with word vector I/O."""

import os
import pathlib
from contextlib import contextmanager
from typing import Tuple, Iterable, Union, BinaryIO, IO, Callable
import numpy as np
from file_or_name import file_or_name
from word_vectors import Vocab, FileType


# The characters we define as "non-binary" when guessing if a file is binary.
ASCII_CHARACTERS = b"".join(map(lambda x: bytes((x,)), range(32, 127))) + b"\n\r\t\f\b"


def find_space(buf: bytes, offset: int) -> Tuple[str, int]:
    """Find the first space starting from offset and return word that spans the spaces and the new offset.

    Args:
        buf: The bytes buffer we are looking for a space in.
        offset: Where in the buffer we start looking.

    Returns:
        A (word, offset) tuple where word is the text (decoded from ``utf-8``) starting at
        the original offset until the first space. Offset is index of the location just
        after the space we just found.
    """
    i = offset + 1
    while buf[i : i + 1] != b" ":
        i += 1
    word = buf[offset:i].decode("utf-8")
    return word, i + 1


@file_or_name(f="rb")
def is_binary(
    f: Union[str, BinaryIO], block_size: int = 512, ratio: float = 0.30, text_characters: bytes = ASCII_CHARACTERS
) -> bool:
    """Guess if a file is binary or not.

    This is based on the implementation from `here`_

    .. _here: https://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python

    Args:
        f: The file we are testing.
        block_size: The amount of the file to read in for checking.
        ratio: How many non-ascii characters before we assume it is binary.
        text_characters: Characters that we define as text characters, the ratio of these characters
            to others is used to determine if the file was binary or not.

    Returns:
        True if the file is binary, False otherwise
    """
    # Because we are operating on an open file object we need to reset where we read from in case
    # people are going to start reading from it right away.
    with bookmark(f):
        block = f.read(block_size)
    # If there are null bytes then it must be binary
    if b"\x00" in block:
        return True
    # We are defining an empty file as a text file.
    elif not block:
        return False

    # Delete all the characters from `text_characters` to leave only the non_text ones
    non_text = block.translate(None, text_characters)
    # If there are more than ratio non-text characters we are a binary file.
    return len(non_text) / len(block) > ratio


@contextmanager
def bookmark(f: IO):
    """Bookmark where we are in a file so we can return.

    This is a context manager that lets us save our spot in an open file,
    to some operations on that file, and then return to the original stop.

    This is very useful for things like sniffing a file. If the file is
    already open and you read in some bytes to estimate the format you need
    to remember to reset to the start or else you will get wrong results.
    This context manager automates this. ::

        f.tell()
        >>> 120
        with bookmark(f):
            _ = f.read(1024)
            print(f.tell())
        >>> 1144
        f.tell()
        >>> 120

    Args:
        f: The file we are bookmarking.
    """
    start = f.tell()
    yield
    f.seek(start)


def to_vocab(words: Iterable[str]) -> Vocab:
    """Convert a series of words to a vocab mapping strings to ints.

    Args:
        words: The words in the vocab

    Returns:
        The Vocabulary
    """
    return {w: i for i, w in enumerate(words)}


def create_output_path(path: Union[str, IO, pathlib.PurePath], file_type: FileType) -> str:
    """Create the output path by stripping the extension and added a new one based on the vector format.

    Args:
        path: The path to the input file.
        file_type: The vector format we are converting to.

    Returns:
        The new output path with an extension determined by the file type.
    """
    if isinstance(path, (str, pathlib.PurePath)):
        path = str(path)
    else:
        path = path.name
    base, _ = os.path.splitext(path)
    return f"{base}.{file_type}"


def uniform_initializer(unif: float) -> Callable[[int], np.ndarray]:
    """Create a vector initialization function that takes a vector size as input.

    Args:
        unif: The bounds that the new vector will be initialized within

    Returns:
        A function that returns a uniformly random vector between ``-unif`` and ``unif``,
    """

    def _unif_initializer(vector_size: int) -> np.ndarray:
        return np.random.rand(-unif, unif, size=(vector_size,))

    return _unif_initializer
