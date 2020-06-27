import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Produce a file size table from our data")
    parser.add_argument("--stats", default="pre-trained.csv")
    parser.add_argument("--reads", default="pre-trained-file-read.csv")
    args = parser.parse_args()

    embed_stats = pd.read_csv(args.stats)
    embed_read = pd.read_csv(args.reads)

    df = embed_stats.merge(embed_read, on=["embedding_id"])[["name", "format", "time"]]

    df = df.groupby(["name", "format"]).agg(["mean", "std", "min", "max"])

    print(df)


if __name__ == "__main__":
    main()
