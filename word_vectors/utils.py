from contextlib import contextmanager
from typing import Tuple, Iterable, Union, BinaryIO, IO
from file_or_name import file_or_name


ASCII_CHARACTERS = b''.join(map(lambda x: bytes((x,)), range(32, 127))) + b'\n\r\t\f\b'


def find_space(m: bytes, offset: int) -> Tuple[str, int]:
    i = offset + 1
    while m[i : i + 1] != b" ":
        i += 1
    word = m[offset:i].decode("utf-8")
    return word, i + 1


def find_max(words: Iterable[str]) -> int:
    """Get the max length of words (as bytes).

    Note:
        This finds the length in bytes, this could be different than the max length of the
        word as returned by `len` because `len` is run on the string objects which might encode
        to more bytes (in utf-8) for example an emoji is often a single character but as bytes it
        could be a few.

    Args:
        words: The series of words.

    Returns:
        The length of the longest word.
    """
    return max(map(len, map(lambda x: x.encode("utf-8"), words)))


@file_or_name(f="rb")
def is_binary(f: Union[str, BinaryIO], block_size: int = 512, ratio: float = 0.30, text_characters: bytes = ASCII_CHARACTERS) -> bool:
    """Guess if a file is binary or not, based on the implementation from here:
        https://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python

    Args:
        f: The file we are testing.
        block_size: The amount of the file to read in for checking.
        ratio: How many non-ascii characters before we assume it is binary.

    Returns:
        True if the file is binary, False otherwise
    """
    # Because we are operating on an open file object we need to reset where we read from in case
    # people are going to start reading from it right away.
    with bookmark(f):
        block = f.read(block_size)
    # If there are null bytes then it must be binary
    if b'\x00' in block:
        return True
    # We are defining an empty file as a text file.
    elif not block:
        return False

    # Delete all the characters from `text_characters` to leave only the non_text ones
    non_text = block.translate(None, text_characters)
    # If there are more than ratio non-text characters we are a binary file.
    return len(non_text) / len(block) > ratio


@contextmanager
def bookmark(f: IO) -> None:
    start = f.tell()
    yield
    f.seek(start)
