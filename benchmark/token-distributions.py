import argparse
from itertools import chain
from collections import Counter
from typing import Dict, Counter as CounterType
import numpy as np
import matplotlib.pyplot as plt
from word_vectors import FileType
from word_vectors.read import read


def wasted_bytes(x: CounterType[int]) -> int:
    longest = max(x)
    wasted = 0
    for k, v in x.items():
        wasted += (longest - k) * v
    return wasted


def total_bytes(x: CounterType[int]) -> int:
    longest = max(x)
    return longest * sum(x.values())


def mean(x: CounterType[int]) -> float:
    return sum(k * v for k, v in x.items()) / sum(x.values())


def std(x: CounterType[int], ddof: int = 1) -> float:
    return np.std(list(chain(*([k] * v for k, v in x.items()))), ddof=ddof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding")
    parser.add_argument("--format", type=FileType.from_string)
    args = parser.parse_args()

    v, _ = read(args.embedding, args.format)

    lengths = Counter(len(token.encode("utf-8")) for token in v)

    print(f"Shortest token: {min(lengths)}")
    print(f"Longest token: {max(lengths)}")
    print(f"Average token length: {mean(lengths)}")
    print(f"Std of token length: {std(lengths)}")
    print(f"Percentage of padding bytes: {wasted_bytes(lengths) / total_bytes(lengths) * 100:.4f}%")


if __name__ == "__main__":
    main()
