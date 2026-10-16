from pathlib import Path
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
TEXT_DIR = RESULTS_DIR / "generated_text"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 25
TEMPERATURES = [0.5, 1.0, 1.5]
GENERATE_WORDS = 100

SEEDS = [
    "there was something in his face that made one trust him",
    "the portrait had taught him to love his own beauty",
    "harry looked at him with a strange expression"
]


def load_tokenizer():
    with open(PROCESSED_DIR / "tokenizer.pkl", "rb") as f:
        return pickle.load(f)


def load_model(name):
    return tf.keras.models.load_model(MODEL_DIR / f"{name}.keras")


def sample_with_temperature(predictions, temperature):
    predictions = np.asarray(predictions).astype("float64")

    predictions = np.log(predictions + 1e-10) / temperature
    exp_predictions = np.exp(predictions)

    probabilities = exp_predictions / np.sum(exp_predictions)

    return np.random.choice(
        len(probabilities),
        p=probabilities
    )


def prepare_seed(seed_text, tokenizer):
    sequence = tokenizer.texts_to_sequences([seed_text])[0]

    if len(sequence) > SEQUENCE_LENGTH:
        sequence = sequence[-SEQUENCE_LENGTH:]

    if len(sequence) < SEQUENCE_LENGTH:
        sequence = [0] * (SEQUENCE_LENGTH - len(sequence)) + sequence

    return np.array(sequence)


def generate_text(
    model,
    tokenizer,
    seed_text,
    temperature,
    num_words=100
):
    generated_text = seed_text
    current_sequence = prepare_seed(seed_text, tokenizer)

    index_word = tokenizer.index_word

    for _ in range(num_words):
        prediction = model.predict(
            current_sequence.reshape(1, -1),
            verbose=0
        )[0]

        next_index = sample_with_temperature(
            prediction,
            temperature
        )

        if next_index == 0:
            continue

        next_word = index_word.get(next_index, "")

        if not next_word:
            continue

        generated_text += " " + next_word

        current_sequence = np.append(
            current_sequence[1:],
            next_index
        )

    return generated_text


def plot_history(model_name):
    history_path = RESULTS_DIR / f"{model_name}_history.json"

    with open(history_path, "r") as f:
        history = json.load(f)

    # Loss plot
    plt.figure()
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.title(f"{model_name.upper()} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{model_name}_loss.png"
    )
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title(f"{model_name.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{model_name}_accuracy.png"
    )
    plt.close()


def generate_for_model(model_name, tokenizer):
    model = load_model(model_name)

    all_outputs = []

    for seed in SEEDS:
        for temperature in TEMPERATURES:
            generated = generate_text(
                model,
                tokenizer,
                seed,
                temperature,
                GENERATE_WORDS
            )

            result = {
                "model": model_name,
                "seed": seed,
                "temperature": temperature,
                "generated_text": generated
            }

            all_outputs.append(result)

            print("=" * 80)
            print("MODEL:", model_name)
            print("TEMPERATURE:", temperature)
            print("SEED:", seed)
            print()
            print(generated)
            print()

    with open(
        TEXT_DIR / f"{model_name}_generated.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_outputs,
            f,
            indent=2,
            ensure_ascii=False
        )


def main():
    tokenizer = load_tokenizer()

    models = [
        "simplernn",
        "birnn",
        "lstm",
        "gru"
    ]

    for model_name in models:
        print()
        print("=" * 80)
        print("PROCESSING", model_name.upper())
        print("=" * 80)

        plot_history(model_name)
        generate_for_model(model_name, tokenizer)

    print()
    print("=" * 80)
    print("GENERATION AND PLOTTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()