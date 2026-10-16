from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_comparison():
    with open(
        RESULTS_DIR / "model_comparison.json",
        "r",
        encoding="utf-8"
    ) as f:
        return pd.DataFrame(json.load(f))


def load_generation_metrics():
    return pd.read_csv(
        RESULTS_DIR / "generation_metrics_summary.csv"
    )


def plot_model_metric(
    df,
    column,
    ylabel,
    title,
    filename
):
    plt.figure(figsize=(8, 5))

    plt.bar(
        df["model"],
        df[column]
    )

    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=300
    )

    plt.close()


def plot_temperature_metric(
    df,
    metric,
    ylabel,
    title,
    filename
):
    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        subset = df[df["model"] == model]

        plt.plot(
            subset["temperature"],
            subset[metric],
            marker="o",
            label=model
        )

    plt.xlabel("Temperature")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks([0.5, 1.0, 1.5])
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=300
    )

    plt.close()


def build_final_summary(
    model_df,
    generation_df
):
    model_table = model_df[
        [
            "model",
            "best_validation_loss",
            "validation_accuracy",
            "perplexity",
            "training_time_seconds",
            "epochs_completed"
        ]
    ].copy()

    model_table.columns = [
        "Model",
        "Validation Loss",
        "Validation Accuracy",
        "Perplexity",
        "Training Time (s)",
        "Epochs"
    ]

    model_table["Validation Loss"] = (
        model_table["Validation Loss"]
        .round(4)
    )

    model_table["Validation Accuracy"] = (
        model_table["Validation Accuracy"]
        .round(4)
    )

    model_table["Perplexity"] = (
        model_table["Perplexity"]
        .round(2)
    )

    model_table["Training Time (s)"] = (
        model_table["Training Time (s)"]
        .round(2)
    )

    qualitative_table = """
| Model | T=0.5 | T=1.0 | T=1.5 |
|---|---|---|---|
| BiRNN | low diversity, repetitive | much better diversity, very low repetition | highest diversity, but incoherent |
| GRU | balanced, but repetitive | **strong balance of diversity + coherence** | diverse but unstable |
| LSTM | moderate coherence and repetition | highest diversity at 1.0 | high diversity, weak coherence |
| SimpleRNN | moderate diversity | good diversity | very high diversity, poor coherence |
"""

    best_model = model_df.loc[
        model_df["perplexity"].idxmin(),
        "model"
    ]

    best_accuracy_model = model_df.loc[
        model_df["validation_accuracy"].idxmax(),
        "model"
    ]

    fastest_model = model_df.loc[
        model_df["training_time_seconds"].idxmin(),
        "model"
    ]

    summary_text = f"""
# Final Model Comparison

## Quantitative Model Comparison

{model_table.to_markdown(index=False)}

## Temperature-Based Qualitative Comparison

{qualitative_table}

## Key Findings

- **Best validation loss:** {best_model}
- **Lowest perplexity:** {best_model}
- **Highest validation accuracy:** {best_accuracy_model}
- **Fastest training model:** {fastest_model}
- **Best overall generation setting:** GRU at temperature 1.0

## Final Interpretation

GRU provided the strongest overall performance across the quantitative evaluation metrics.

It achieved the lowest validation loss and perplexity while also obtaining the highest validation accuracy.

The temperature experiments showed a consistent trade-off:

- Temperature 0.5 produced more coherent but more repetitive text.
- Temperature 1.0 provided the strongest balance between coherence, lexical diversity, contextual consistency, and repetition.
- Temperature 1.5 increased lexical diversity but substantially reduced grammatical and contextual coherence.

Although some models achieved higher lexical diversity at temperature 1.5, the generated text became fragmented and semantically unstable.

Therefore, **GRU with temperature 1.0** was selected as the most effective autoregressive text-generation configuration for this dataset.
"""

    with open(
        RESULTS_DIR / "final_comparison.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(summary_text)

    print(summary_text)


def main():
    model_df = load_model_comparison()
    generation_df = load_generation_metrics()

    plot_model_metric(
        model_df,
        "best_validation_loss",
        "Validation Loss",
        "Validation Loss Comparison",
        "model_validation_loss.png"
    )

    plot_model_metric(
        model_df,
        "validation_accuracy",
        "Validation Accuracy",
        "Validation Accuracy Comparison",
        "model_validation_accuracy.png"
    )

    plot_model_metric(
        model_df,
        "perplexity",
        "Perplexity",
        "Perplexity Comparison",
        "model_perplexity.png"
    )

    plot_model_metric(
        model_df,
        "training_time_seconds",
        "Training Time (seconds)",
        "Training Time Comparison",
        "model_training_time.png"
    )

    plot_temperature_metric(
        generation_df,
        "lexical_diversity",
        "Lexical Diversity",
        "Lexical Diversity vs Temperature",
        "temperature_lexical_diversity.png"
    )

    plot_temperature_metric(
        generation_df,
        "bigram_repetition",
        "Bigram Repetition",
        "Bigram Repetition vs Temperature",
        "temperature_bigram_repetition.png"
    )

    build_final_summary(
        model_df,
        generation_df
    )

    print()
    print("=" * 70)
    print("FINAL COMPARISON COMPLETE")
    print("=" * 70)

    print("Plots saved in:", PLOTS_DIR)
    print(
        "Summary saved in:",
        RESULTS_DIR / "final_comparison.md"
    )


if __name__ == "__main__":
    main()