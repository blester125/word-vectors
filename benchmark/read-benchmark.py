import time
import json
import argparse
from word_vectors import FileType
from word_vectors.read import read_glove, read_w2v_text, read_w2v, read_dense


def main():
    parser = argparse.ArgumentParser(description="Time how long it takes to read an embedding file.")
    parser.add_argument("embedding", help="The path to the embedding file to read")
    parser.add_argument(
        "--format", required=True, type=FileType.from_string, help="The file format we are benchmarking"
    )
    args = parser.parse_args()

    if args.format is FileType.GLOVE:
        reader = read_glove
    elif args.format is FileType.W2V_TEXT:
        reader = read_w2v_text
    elif args.format is FileType.W2V:
        reader = read_w2v
    elif args.format is FileType.DENSE:
        reader = read_dense
    else:
        raise ValueError(f"Unknown file format, got {args.format}")

    tic = time.time()
    reader(args.embedding)
    toc = time.time()

    print(json.dumps({"file": args.embedding, "format": str(args.format), "time": toc - tic}))


if __name__ == "__main__":
    main()
