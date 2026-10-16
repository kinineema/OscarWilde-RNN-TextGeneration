from pathlib import Path
import json
import pandas as pd


TEXT_DIR = Path("results/generated_text")
RESULTS_DIR = Path("results")


def repetition_rate(words):
    if len(words) == 0:
        return 0.0

    unique_words = len(set(words))
    return 1 - (unique_words / len(words))


def lexical_diversity(words):
    if len(words) == 0:
        return 0.0

    return len(set(words)) / len(words)


def bigram_repetition(words):
    if len(words) < 2:
        return 0.0

    bigrams = [
        (words[i], words[i + 1])
        for i in range(len(words) - 1)
    ]

    unique_bigrams = len(set(bigrams))

    return 1 - (unique_bigrams / len(bigrams))


def evaluate_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for sample in data:
        text = sample["generated_text"]
        words = text.lower().split()

        rows.append({
            "model": sample["model"],
            "seed": sample["seed"],
            "temperature": sample["temperature"],
            "word_count": len(words),
            "lexical_diversity": lexical_diversity(words),
            "repetition_rate": repetition_rate(words),
            "bigram_repetition": bigram_repetition(words)
        })

    return rows


def main():
    all_rows = []

    for path in TEXT_DIR.glob("*_generated.json"):
        all_rows.extend(
            evaluate_file(path)
        )

    df = pd.DataFrame(all_rows)

    df.to_csv(
        RESULTS_DIR / "generation_metrics.csv",
        index=False
    )

    summary = (
        df.groupby(
            ["model", "temperature"]
        )[
            [
                "lexical_diversity",
                "repetition_rate",
                "bigram_repetition"
            ]
        ]
        .mean()
        .round(4)
    )

    print("=" * 70)
    print("GENERATION QUALITY METRICS")
    print("=" * 70)

    print(summary)

    summary.to_csv(
        RESULTS_DIR / "generation_metrics_summary.csv"
    )


if __name__ == "__main__":
    main()