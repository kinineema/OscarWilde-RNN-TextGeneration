# Autoregressive Literary Text Generation using Recurrent Neural Networks

## A Comparative Study on Oscar Wilde's *The Picture of Dorian Gray*

This project implements and compares four recurrent neural network architectures for **word-level autoregressive text generation**:

- Simple RNN
- Bidirectional RNN (BiRNN)
- LSTM
- GRU

All four models are trained under comparable experimental conditions on Oscar Wilde's *The Picture of Dorian Gray* to predict the next word in a sequence and generate text autoregressively.

The project also studies how different **sampling temperatures** affect coherence, diversity, contextual consistency, and repetition in generated text.

---

## 1. Objective

The main objectives of this project are:

1. Prepare a literary text dataset for autoregressive next-word prediction.
2. Construct fixed-length input-target sequences.
3. Implement Simple RNN, Bidirectional RNN, LSTM, and GRU models.
4. Train all models under comparable experimental conditions.
5. Evaluate the models using:
   - Validation loss
   - Next-word prediction accuracy
   - Perplexity
   - Training time
6. Generate text using the same seed sequences.
7. Analyse text generation at temperatures:
   - 0.5
   - 1.0
   - 1.5
8. Compare generated text based on:
   - Coherence
   - Lexical diversity
   - Contextual consistency
   - Repetition
9. Identify the most effective model and sampling temperature.

---

## 2. Dataset

### The Picture of Dorian Gray

**Author:** Oscar Wilde

The project uses the public-domain text of *The Picture of Dorian Gray* obtained from Project Gutenberg.

Dataset source:

https://www.gutenberg.org/ebooks/174

### Dataset Characteristics

| Property | Value |
|---|---:|
| Raw characters | 448,657 |
| Cleaned characters | 427,636 |
| Total tokens | 79,759 |
| Vocabulary size | 6,914 |
| Sequence length | 25 |
| Total input-target sequences | 79,734 |
| Training sequences | 63,787 |
| Validation sequences | 15,947 |

The dataset provides a mixture of narrative passages, dialogue, descriptions, and character interactions, making it suitable for studying literary next-word generation.

---

## 3. Text Preprocessing

The preprocessing pipeline performs the following operations:

- Removes Project Gutenberg header and footer content.
- Converts text to lowercase.
- Normalizes quotation marks and punctuation.
- Normalizes whitespace.
- Performs word-level tokenization.
- Creates a vocabulary from the corpus.
- Converts words to integer token IDs.
- Constructs autoregressive input-target sequences.

Each sample contains:

- **Input:** Previous 25 words
- **Target:** Next word

Example:

```text
Input:
the portrait had taught him to love his own

Target:
beauty
```

The same processed dataset is used for all four neural-network architectures to ensure a fair comparison.

---

## 4. Train-Validation Split

The complete dataset contains:

```text
79,734 sequences
```

An 80:20 train-validation split is used:

```text
Training sequences   : 63,787
Validation sequences : 15,947
```

A fixed random seed is used to improve experimental reproducibility.

---

## 5. Models Implemented

### Simple RNN

The Simple RNN provides the baseline recurrent architecture.

Architecture:

```text
Embedding
   ↓
SimpleRNN (128 units)
   ↓
Dropout (0.2)
   ↓
Dense Softmax
```

Simple RNNs are computationally efficient but can struggle with long-term dependencies because of vanishing-gradient effects.

---

### Bidirectional RNN

The Bidirectional RNN uses recurrent processing in both directions during training.

Architecture:

```text
Embedding
   ↓
Bidirectional SimpleRNN
   ↓
Dropout (0.2)
   ↓
Dense Softmax
```

This architecture provides richer contextual representations during training, although bidirectionality does not necessarily guarantee better autoregressive generation.

---

### LSTM

Long Short-Term Memory networks introduce memory cells and gating mechanisms to better preserve long-range information.

Architecture:

```text
Embedding
   ↓
LSTM (128 units)
   ↓
Dropout (0.2)
   ↓
Dense Softmax
```

LSTMs are designed to reduce the vanishing-gradient problem associated with conventional recurrent networks.

---

### GRU

The Gated Recurrent Unit provides gated memory similar to LSTM while using a simpler recurrent structure.

Architecture:

```text
Embedding
   ↓
GRU (128 units)
   ↓
Dropout (0.2)
   ↓
Dense Softmax
```

GRUs often provide a good compromise between modelling capability and computational complexity.

---

## 6. Common Experimental Configuration

To ensure a fair comparison, all four models use the same experimental configuration.

| Parameter | Value |
|---|---:|
| Sequence length | 25 |
| Embedding dimension | 128 |
| Recurrent units | 128 |
| Dropout | 0.2 |
| Batch size | 128 |
| Maximum epochs | 20 |
| Validation split | 20% |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Early stopping patience | 3 |
| Random seed | 42 |

Early stopping is used to prevent unnecessary training once validation performance stops improving.

---

## 7. Quantitative Model Comparison

| Model | Validation Loss | Validation Accuracy | Perplexity | Training Time (s) | Epochs |
|---|---:|---:|---:|---:|---:|
| SimpleRNN | 5.7324 | 0.1227 | 308.70 | **63.15** | 7 |
| BiRNN | 5.9382 | 0.0971 | 379.23 | 71.06 | 5 |
| LSTM | 5.7811 | 0.1248 | 324.10 | 236.31 | 10 |
| **GRU** | **5.6830** | **0.1338** | **293.82** | 149.57 | 8 |

### Quantitative Findings

The **GRU** achieved:

- Lowest validation loss: **5.6830**
- Highest validation accuracy: **13.38%**
- Lowest perplexity: **293.82**

The **Simple RNN** was the fastest model to train at approximately **63 seconds**.

The LSTM required the longest training time, while the Bidirectional RNN obtained the weakest validation performance.

---

## 8. Perplexity

Perplexity is derived from cross-entropy loss using:

```text
Perplexity = exp(Loss)
```

Lower perplexity indicates greater confidence in next-word prediction.

The GRU achieved the lowest perplexity:

```text
GRU Perplexity = 293.82
```

making it the strongest model according to this metric.

---

## 9. Text Generation

Text is generated autoregressively.

Starting from a seed sequence:

1. The model predicts a probability distribution for the next word.
2. A word is sampled from this distribution.
3. The predicted word is appended to the sequence.
4. The most recent 25 tokens are passed back to the model.
5. The process repeats until the required number of words has been generated.

Three common seed sequences are used for all four models.

Examples include:

```text
there was something in his face that made one trust him
```

```text
the portrait had taught him to love his own beauty
```

```text
harry looked at him with a strange expression
```

---

## 10. Sampling Temperature

Sampling temperature controls the randomness of next-word selection.

Three temperatures are evaluated:

```text
0.5
1.0
1.5
```

### Temperature 0.5

The probability distribution becomes sharper.

This generally results in:

- more predictable words
- better local coherence
- lower lexical diversity
- greater repetition

### Temperature 1.0

The original probability distribution is retained.

This generally gives the strongest balance between:

- coherence
- diversity
- contextual consistency
- repetition

### Temperature 1.5

The distribution becomes flatter.

This results in:

- greater lexical diversity
- more surprising word choices
- less repetition
- reduced grammatical consistency
- weaker contextual coherence

---

## 11. Temperature-Based Qualitative Comparison

| Model | T=0.5 | T=1.0 | T=1.5 |
|---|---|---|---|
| BiRNN | low diversity, repetitive | much better diversity, very low repetition | highest diversity, but incoherent |
| GRU | balanced, but repetitive | **strong balance of diversity + coherence** | diverse but unstable |
| LSTM | moderate coherence and repetition | highest diversity at 1.0 | high diversity, weak coherence |
| SimpleRNN | moderate diversity | good diversity | very high diversity, poor coherence |

---

## 12. Generated-Text Metrics

Generated text is additionally analysed using:

### Lexical Diversity

```text
Unique words / Total words
```

Higher values indicate a wider vocabulary within the generated sample.

### Repetition Rate

Measures how frequently already-used words are repeated.

### Bigram Repetition

Measures repetition of two-word combinations.

The experiments show a clear relationship between sampling temperature and diversity.

At temperature **0.5**, all models tend to be more repetitive.

At temperature **1.0**, lexical diversity improves substantially while maintaining useful textual structure.

At temperature **1.5**, lexical diversity becomes very high, but human inspection shows that grammatical and semantic coherence deteriorates.

Therefore, lexical diversity alone should not be interpreted as generation quality.

---

## 13. Temperature and Lexical Diversity

Average lexical-diversity values:

| Model | T=0.5 | T=1.0 | T=1.5 |
|---|---:|---:|---:|
| BiRNN | 0.4376 | 0.6990 | **0.9119** |
| GRU | 0.4834 | 0.7391 | 0.8753 |
| LSTM | 0.4829 | **0.7659** | 0.8450 |
| SimpleRNN | **0.5012** | 0.7478 | 0.9055 |

Increasing temperature clearly increases lexical diversity.

However, generated-text inspection demonstrates that the very high diversity obtained at temperature 1.5 comes at the cost of coherence.

---

## 14. Bigram Repetition

| Model | T=0.5 | T=1.0 | T=1.5 |
|---|---:|---:|---:|
| BiRNN | 0.1749 | 0.0061 | 0.0000 |
| GRU | 0.1870 | 0.0183 | 0.0031 |
| LSTM | 0.1476 | 0.0124 | 0.0000 |
| SimpleRNN | 0.1600 | 0.0092 | 0.0000 |

Bigram repetition falls significantly as temperature increases.

This confirms the expected trade-off between deterministic generation and diversity.

---

## 15. Key Experimental Findings

- **Best validation loss:** GRU
- **Lowest perplexity:** GRU
- **Highest validation accuracy:** GRU
- **Fastest model:** SimpleRNN
- **Most balanced sampling temperature:** 1.0
- **Best overall configuration:** GRU at temperature 1.0

---

## 16. Final Comparison

### Simple RNN

Advantages:

- Fastest training
- Relatively simple architecture
- Capable of learning short-term language structure

Limitations:

- More repetitive generation
- Weaker long-term contextual consistency

### Bidirectional RNN

Advantages:

- Processes richer contextual information during training
- High lexical diversity at higher temperatures

Limitations:

- Highest validation loss
- Highest perplexity
- Lowest validation accuracy
- Generated text often lacks coherent sequential structure

### LSTM

Advantages:

- Better handling of long-term dependencies than Simple RNN
- Good lexical diversity
- Reasonably coherent output at moderate temperature

Limitations:

- Highest training time
- Did not outperform GRU quantitatively

### GRU

Advantages:

- Lowest validation loss
- Lowest perplexity
- Highest validation accuracy
- Strong qualitative generation at moderate temperature
- Lower computational complexity than LSTM

The GRU therefore provided the strongest overall balance of predictive performance and generated-text quality.

---

## 17. Conclusion

This project compared four recurrent neural-network architectures for autoregressive literary text generation using Oscar Wilde's *The Picture of Dorian Gray*.

Among the evaluated models, **GRU achieved the strongest quantitative performance**, with:

```text
Validation Loss     : 5.6830
Validation Accuracy : 0.1338
Perplexity           : 293.82
```

The sampling-temperature experiments demonstrated an important trade-off between coherence and diversity.

A low temperature of **0.5** produced relatively coherent but repetitive text, while a high temperature of **1.5** produced highly diverse but frequently incoherent text.

A temperature of **1.0** offered the most effective compromise between lexical diversity, repetition, and contextual coherence.

Considering both quantitative model performance and qualitative generated-text analysis:

**GRU with a sampling temperature of 1.0 was identified as the best overall autoregressive text-generation configuration.**

---

## 18. Repository Structure

```text
OscarWilde_TextGen/
│
├── data/
│   ├── dorian_gray.txt
│   ├── cleaned_dorian_gray.txt
│   └── processed/
│       ├── X.npy
│       ├── y.npy
│       ├── tokenizer.pkl
│       └── dataset_stats.json
│
├── models/
│   ├── simplernn.keras
│   ├── birnn.keras
│   ├── lstm.keras
│   └── gru.keras
│
├── results/
│   ├── plots/
│   ├── generated_text/
│   ├── model_comparison.json
│   ├── generation_metrics.csv
│   ├── generation_metrics_summary.csv
│   └── final_comparison.md
│
├── src/
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── generate_and_plot.py
│   ├── evaluate_generation.py
│   └── final_comparison.py
│
├── notebook/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 19. Installation

Python 3.11 was used for this project.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 20. Execution

### Step 1 — Prepare Dataset

```powershell
python src\prepare_data.py
```

This creates the cleaned corpus, tokenizer, vocabulary information, and input-target sequences.

### Step 2 — Train Models

```powershell
python src\train_models.py
```

This trains:

- Simple RNN
- Bidirectional RNN
- LSTM
- GRU

and stores the trained models and training histories.

### Step 3 — Generate Text and Training Plots

```powershell
python src\generate_and_plot.py
```

This generates samples for temperatures:

```text
0.5
1.0
1.5
```

and creates model training/validation plots.

### Step 4 — Evaluate Generated Text

```powershell
python src\evaluate_generation.py
```

This calculates:

- lexical diversity
- repetition rate
- bigram repetition

### Step 5 — Generate Final Comparison

```powershell
python src\final_comparison.py
```

This produces final model-comparison plots and the final experimental summary.

---

## 21. Dependencies

Major dependencies include:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- NLTK
- Tabulate

Install all dependencies using:

```powershell
pip install -r requirements.txt
```

---

## 22. Output Files

Training results:

```text
results/model_comparison.json
```

Generated text:

```text
results/generated_text/
```

Generation metrics:

```text
results/generation_metrics.csv
results/generation_metrics_summary.csv
```

Final analysis:

```text
results/final_comparison.md
```

Plots:

```text
results/plots/
```

Trained models:

```text
models/
```

---

## Author
Name: Neema Naveen Kini
Reg No: 230953070
Subject: Foundations of Generative AI (ICT4403)(Assignment 2)