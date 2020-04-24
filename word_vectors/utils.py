from typing import Tuple, Iterable


def _find_space(m: bytes, offset: int) -> Tuple[str, int]:
    i = offset + 1
    while m[i : i + 1] != b" ":
        i += 1
    word = m[offset:i].decode("utf-8")
    return word, i + 1


def _find_max(words: Iterable[str]) -> int:
    """Get the max length of words (as bytes)."""
    return max(map(len, map(lambda x: x.encode("utf-8"), words)))
