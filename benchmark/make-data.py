#!/usr/bin/env python3

import string
import random
import argparse
import numpy as np
from word_vectors import FileType
from word_vectors.write import write


def random_string(length=None, min_: int = 1, max_: int = 20) -> str:
    length = random.randint(min_, max_ - 1) if length is None else length
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def main():
    parser = argparse.ArgumentParser(description="Create Fake Embeddings data for benchmarking.")
    parser.add_argument("--vsz", type=int, required=True)
    parser.add_argument("--dsz", type=int, required=True)
    parser.add_argument("--max", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    words = {}
    while len(words) < args.vsz:
        word = random_string(min_=1, max_=args.max_)
        if word not in words:
            words[word] = len(words)

    vectors = np.random.rand(args.vsz, args.dsz).astype(np.float32)

    for format in FileType:
        write(f"{args.output}.{format}", words, vectors, format)


if __name__ == "__main__":
    main()
