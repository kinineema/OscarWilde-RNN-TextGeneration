from pathlib import Path
import json
import pickle
import time
import math

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Bidirectional,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
EMBEDDING_DIM = 128
HIDDEN_UNITS = 128
VALIDATION_SPLIT = 0.2

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")

    with open(PROCESSED_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1

    return X, y, tokenizer, vocab_size


def split_data(X, y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * (1 - VALIDATION_SPLIT))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    return X_train, X_val, y_train, y_val


def build_simple_rnn(vocab_size, sequence_length):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=sequence_length),
        SimpleRNN(HIDDEN_UNITS),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax")
    ])

    return model


def build_bidirectional_rnn(vocab_size, sequence_length):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=sequence_length),
        Bidirectional(SimpleRNN(HIDDEN_UNITS)),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax")
    ])

    return model


def build_lstm(vocab_size, sequence_length):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=sequence_length),
        LSTM(HIDDEN_UNITS),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax")
    ])

    return model


def build_gru(vocab_size, sequence_length):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=sequence_length),
        GRU(HIDDEN_UNITS),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax")
    ])

    return model


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def train_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val
):
    compile_model(model)

    print()
    print("=" * 60)
    print(f"TRAINING {name}")
    print("=" * 60)

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.time() - start_time

    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    perplexity = math.exp(val_loss)

    model_path = MODEL_DIR / f"{name.lower()}.keras"
    model.save(model_path)

    result = {
        "model": name,
        "best_validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "perplexity": float(perplexity),
        "training_time_seconds": float(training_time),
        "epochs_completed": len(history.history["loss"]),
        "model_path": str(model_path)
    }

    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with open(
        RESULTS_DIR / f"{name.lower()}_history.json",
        "w"
    ) as f:
        json.dump(history_data, f, indent=2)

    return result


def main():
    X, y, tokenizer, vocab_size = load_data()

    X_train, X_val, y_train, y_val = split_data(X, y)

    sequence_length = X.shape[1]

    print("=" * 60)
    print("DATA LOADED")
    print("=" * 60)

    print("Vocabulary size:", vocab_size)
    print("Total sequences:", len(X))
    print("Training sequences:", len(X_train))
    print("Validation sequences:", len(X_val))
    print("Sequence length:", sequence_length)

    models = {
        "SimpleRNN": build_simple_rnn(
            vocab_size,
            sequence_length
        ),

        "BiRNN": build_bidirectional_rnn(
            vocab_size,
            sequence_length
        ),

        "LSTM": build_lstm(
            vocab_size,
            sequence_length
        ),

        "GRU": build_gru(
            vocab_size,
            sequence_length
        )
    }

    all_results = []

    for name, model in models.items():
        result = train_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val
        )

        all_results.append(result)

        tf.keras.backend.clear_session()

    with open(
        RESULTS_DIR / "model_comparison.json",
        "w"
    ) as f:
        json.dump(all_results, f, indent=2)

    print()
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    for result in all_results:
        print()
        print("Model:", result["model"])
        print(
            "Validation Loss:",
            round(result["best_validation_loss"], 4)
        )
        print(
            "Validation Accuracy:",
            round(result["validation_accuracy"], 4)
        )
        print(
            "Perplexity:",
            round(result["perplexity"], 2)
        )
        print(
            "Training Time:",
            round(result["training_time_seconds"], 2),
            "seconds"
        )
        print(
            "Epochs:",
            result["epochs_completed"]
        )


if __name__ == "__main__":
    main()