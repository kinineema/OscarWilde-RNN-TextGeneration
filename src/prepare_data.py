from pathlib import Path
import json
import re
import pickle

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


RAW_PATH = Path("data/dorian_gray.txt")
PROCESSED_DIR = Path("data/processed")

SEQUENCE_LENGTH = 25


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")

    # Remove Project Gutenberg header/footer if present
    start_patterns = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]

    end_patterns = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]

    upper_text = text.upper()

    for pattern in start_patterns:
        idx = upper_text.find(pattern)
        if idx != -1:
            line_end = text.find("\n", idx)
            if line_end != -1:
                text = text[line_end + 1 :]
            break

    upper_text = text.upper()

    for pattern in end_patterns:
        idx = upper_text.find(pattern)
        if idx != -1:
            text = text[:idx]
            break

    # Normalize punctuation and whitespace
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("—", " ")
    text = text.replace("–", " ")

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_sequences(text: str):
    tokenizer = Tokenizer(
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    )

    tokenizer.fit_on_texts([text])

    token_list = tokenizer.texts_to_sequences([text])[0]

    sequences = []

    for i in range(SEQUENCE_LENGTH, len(token_list)):
        sequence = token_list[i - SEQUENCE_LENGTH : i + 1]
        sequences.append(sequence)

    sequences = np.array(sequences, dtype=np.int32)

    X = sequences[:, :-1]
    y = sequences[:, -1]

    return tokenizer, X, y


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}. "
            "Save The Picture of Dorian Gray as data/dorian_gray.txt"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_text = RAW_PATH.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    cleaned_text = clean_text(raw_text)

    tokenizer, X, y = build_sequences(cleaned_text)

    vocab_size = len(tokenizer.word_index) + 1

    stats = {
        "title": "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "sequence_length": SEQUENCE_LENGTH,
        "raw_characters": len(raw_text),
        "cleaned_characters": len(cleaned_text),
        "total_tokens": len(
            tokenizer.texts_to_sequences([cleaned_text])[0]
        ),
        "vocabulary_size": vocab_size,
        "training_sequences": len(X),
    }

    Path("data/cleaned_dorian_gray.txt").write_text(
        cleaned_text,
        encoding="utf-8",
    )

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)

    with open(
        PROCESSED_DIR / "tokenizer.pkl",
        "wb",
    ) as handle:
        pickle.dump(tokenizer, handle)

    with open(
        PROCESSED_DIR / "dataset_stats.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(stats, handle, indent=2)

    print("=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print()
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()