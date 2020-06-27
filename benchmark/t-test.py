#!/usr/bin/env python3

import os
import json
import argparse
from scipy import stats


def less_than_reject_null(t, p, alpha):
    if p / 2.0 < alpha and t < 0:
        return True
    return False


def greater_than_reject_null(t, p, alpha):
    if p / 2.0 < alpha and t > 0:
        return True
    return False


def t_test(exp, alpha):
    t, p = stats.ttest_ind_from_stats(
        exp["treatment"]["mean"],
        exp["treatment"]["std"],
        exp["treatment"]["obs"],
        exp["control"]["mean"],
        exp["control"]["std"],
        exp["control"]["obs"],
    )
    if exp["test_type"] == "greater":
        reject_null = greater_than_reject_null(t, p, alpha)
    else:
        reject_null = less_than_reject_null(t, p, alpha)
    return {
        "experiment": exp["experiment"],
        "t": t,
        "p": p,
        "result": "reject" if reject_null else "fail to reject",
        "research_hypo": exp["research_hypo"],
        "null_hypo": exp["null_hypo"],
    }


def print_result(result):
    print(f"Statistically Significant: {result['experiment']}")
    print(f"\tt: {result['t']}")
    print(f"\tp: {result['p']}")
    if result["result"] == "reject":
        print(f"\tWe reject the null hypothesis, therefore:\n\t\t \"{result['research_hypo']}\"")
    else:
        print(f"\tWe cannot reject the null hypothesis, therefore:\n\t\t \"{result['null_hypo']}\"")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.experiments) as f:
        experiments = json.load(f)

    if not isinstance(experiments, list):
        experiments = []

    results = [t_test(exp, args.alpha) for exp in experiments]
    for result in results:
        print_result(result)

    args.output = os.path.splitext(args.experiments)[0] + "-results.json" if args.output is None else args.output
    with open(args.output, "w") as wf:
        json.dump(results, wf, indent=2)


if __name__ == "__main__":
    main()
