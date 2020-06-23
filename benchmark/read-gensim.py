#!/usr/bin/env python3

import time
import json
import argparse
from gensim.models.keyedvectors import Word2VecKeyedVectors


def main():
    parser = argparse.ArgumentParser(description="Time how long it takes gensim to read an embedding file.")
    parser.add_argument("embedding", help="The path ot the embeddings file to read")
    parser.add_argument("--format", required=True, choices=("binary", "text"))
    args = parser.parse_args()

    binary = True if args.format == "binary" else False

    tic = time.time()
    _ = Word2VecKeyedVectors.load_word2vec_format(args.embedding, binary=binary)
    toc = time.time()

    print(json.dumps({"file": args.embedding, "format": args.format, "time": toc - tic}))


if __name__ == "__main__":
    main()
