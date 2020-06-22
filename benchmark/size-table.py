import math
import argparse
from typing import Optional
import pandas as pd


DISK_SIZE = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
MEM_SIZE = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")


def human_size(num_bytes: int, unit: Optional[str] = None, divisor: int = 1024, floatfmt: str = ".4f") -> str:
    if num_bytes == 0:
        return "0B"
    size_names = DISK_SIZE if divisor == 1000 else MEM_SIZE
    if unit is None:
        i = int(math.floor(math.log(num_bytes, divisor)))
    else:
        i = size_names.index(unit)
    p = math.pow(divisor, i)
    s = num_bytes / p
    return f"{s:{floatfmt}} {size_names[i]}"


def main():
    parser = argparse.ArgumentParser(description="Produce a file size table from our data")
    parser.add_argument("--stats", default="pre-trained.csv")
    parser.add_argument("--sizes", default="pre-trained-file-size.csv")
    parser.add_argument("--unit", choices=("GiB", "MiB"), default=None)
    args = parser.parse_args()

    embed_stats = pd.read_csv(args.stats)
    embed_size = pd.read_csv(args.sizes)

    df = embed_stats.merge(embed_size, on=["embedding_id"])[["name", "format", "bytes"]]

    df.bytes = df.bytes.apply(lambda x: human_size(x, args.unit))

    print(df)


if __name__ == "__main__":
    main()
