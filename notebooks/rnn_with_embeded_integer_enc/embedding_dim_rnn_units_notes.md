# `embedding_dimension` and `rnn_units` — Summary Notes

## What they are
Both are **hyperparameters** — chosen by you, not learned by training — that
control the *size/capacity* of two layers.

```python
embedding_dimension = 16
rnn_units = 16
```

| Parameter | Controls | Shape it defines |
|---|---|---|
| `embedding_dimension` | numbers used to represent **each word** | `Embedding` lookup table = `(vocab_size, 16)` |
| `rnn_units` | size of the RNN's **memory / hidden state** | hidden state vector = `(16,)`, updated every timestep |

## Data flow: how they connect

```mermaid
flowchart LR
    A["Word ID (int)<br/>e.g. 13 = 'love'"] --> B["Embedding lookup table<br/>(vocab_size x 16)"]
    B --> C["Word vector<br/>size = embedding_dimension (16)"]
    C --> D["SimpleRNN<br/>processes one word at a time"]
    D --> E["Hidden state updated each step<br/>size = rnn_units (16)"]
    E --> F["Final hidden state<br/>(16 numbers = sentence summary)"]
    F --> G["Dense(1, sigmoid)"]
    G --> H["Probability (0-1)"]
```

## Why size matters (capacity vs. overfitting trade-off)

```mermaid
flowchart TD
    S{"Choose dimension size"}
    S -->|"Too small"| U["Underfitting<br/>not enough room to capture patterns"]
    S -->|"Too large"| O["Overfitting<br/>too many params for small data,<br/>memorizes instead of generalizes"]
    S -->|"Just right"| G["Good fit<br/>matches dataset size/complexity"]
```

## Why `16` was chosen here
Dataset = 20 sentences, 31 unique words → very small.

```mermaid
flowchart LR
    A["vocab_size = 32"] --> B["embedding params = 32 x 16 = 512"]
    C["train examples = 16 sentences"] -.->|"already comparable to param count"| B
    B --> D["Bigger dims (e.g. 256) would\nmassively overfit this tiny data"]
```

`16`/`16` is simply a **small, sensible starting point** matched to a toy-scale
dataset — not a value with special significance.

## Practical sizing guide

| Situation | Typical `embedding_dimension` / `rnn_units` |
|---|---|
| Toy/tiny dataset (this notebook) | 8 – 32 |
| Small real dataset (hundreds–thousands of examples) | 32 – 128 |
| Large real dataset (e.g. IMDB, 25k+ reviews) | 64 – 300 |

## One-line takeaway
> Both parameters trade off **model capacity** against **overfitting risk** —
> pick them relative to how much data and vocabulary complexity you actually
> have, then tune based on validation performance.
