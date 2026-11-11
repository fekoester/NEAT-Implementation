"""Plot training curves from neat_train's stats.csv files.

    python scripts/plot_stats.py results/dodge/stats.csv --title "dodge" --out docs/media/dodge_training.png
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--ylabel", default="fitness")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.stats)))
    gen = [int(r["generation"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]  # noqa: E731

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(gen, col("best_fitness"), label="generation best", color="tab:blue", lw=1)
    ax1.plot(gen, col("mean_fitness"), label="population mean", color="tab:orange", lw=1.5)
    ax1.set_xlabel("generation")
    ax1.set_ylabel(args.ylabel)
    ax1.set_title(args.title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(gen, col("species"), label="species", color="tab:green")
    ax2.plot(gen, col("mean_hidden"), label="mean hidden nodes", color="tab:purple")
    ax2.set_xlabel("generation")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title("speciation and complexification")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
