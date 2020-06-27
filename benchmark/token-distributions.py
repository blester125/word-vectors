import os
import json
import argparse
from itertools import chain
from collections import Counter
from typing import Dict, Counter as CounterType
import numpy as np
import matplotlib.pyplot as plt
from word_vectors import FileType
from word_vectors.read import read


def mean(x: CounterType[int]) -> float:
    return sum(k * v for k, v in x.items()) / sum(x.values())


def std(x: CounterType[int], ddof: int = 1) -> float:
    return np.std(list(chain(*([k] * v for k, v in x.items()))), ddof=ddof)


def plot_distribution(x: CounterType[int], title: str = "Type length distribution", output: str = "distribution.png"):
    norm = sum(x.values())
    x = Counter({k: x[k] / norm for k in range(max(x))})
    plt.bar(range(max(x)), [x[k] for k in range(max(x))], width=1)
    plt.title(title)
    plt.xlabel("Type Length")
    plt.ylabel("Proportion of vocabulary")
    plt.savefig(output)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding")
    parser.add_argument("--embed-name", "--embed_name", help="Force a name if the file name isn't descriptive")
    parser.add_argument("--format", type=FileType.from_string)
    parser.add_argument(
        "--output",
        help="The name of the output files sans extensions. Will create a .json or stats and a .png the the distirbution graph",
    )
    args = parser.parse_args()

    v, wv = read(args.embedding, args.format)

    lengths = Counter(len(token.encode("utf-8")) for token in v)

    vsz = len(v)
    dsz = wv.shape[-1]
    min_l = min(lengths)
    max_l = max(lengths)
    avg_l = mean(lengths)
    std_l = std(lengths)

    print(f"Vocab size: {vsz}")
    print(f"Vector size: {dsz}")
    print(f"Shortest token: {min_l}")
    print(f"Longest token: {max_l}")
    print(f"Average token length: {avg_l}")
    print(f"Std of token length: {std_l}")

    embed_name = os.path.splitext(os.path.basename(args.embedding))[0] if args.embed_name is None else args.embed_name

    data = {
        "name": embed_name,
        "stats": {
            "vocab_size": vsz,
            "vector_size": dsz,
            "min_length": min_l,
            "max_length": max_l,
            "avg_length": avg_l,
            "std_length": std_l,
        },
        "counts": {**lengths},
    }

    if args.output is None:
        args.output = embed_name

    with open(f"{args.output}.json", "w") as wf:
        json.dump(data, wf, indent=2)

    plot_distribution(lengths, title=f"Type Length Distribution for {embed_name}", output=f"{args.output}.png")


if __name__ == "__main__":
    main()
