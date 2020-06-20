import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Create a graph from the benchmarking data.")
    parser.add_argument("--data", help="The CSV with the benchmark data.", required=True)
    parser.add_argument("--y", help="The dependent variable", required=True)
    parser.add_argument("--x", help="The independent variable", required=True)
    parser.add_argument("--y-text", help="Optional pretty text for the y variable")
    parser.add_argument("--x-text", help="Optional pretty text for the x variable")
    parser.add_argument("--filter", help="Columns to filter from the group by", nargs="+", default=[])

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.drop(args.filter, axis=1)

    group_by = sorted(list(set(df.columns) - {args.y, args.x}))

    groups = df.groupby(group_by)

    for _, group in groups:
        # print(group[[args.x, args.y]])
        print(group)
        print("=" * 80)


if __name__ == "__main__":
    main()
