# Types of Language Models

Language Models (LMs) have evolved through **three major eras**, each solving the limitations of the previous one. The journey moves from *counting words* → *learning sequences* → *understanding context at scale*.

```
Stats Era  →  Neural Era  →  Transformer Era (LLMs)
 (1990s)       (2010s)             (2017 → today)
```

---

## 1. Stats Era — N-grams & Markov Chains

The earliest language models were **purely statistical**. They estimated the probability of the next word by **counting how often word sequences appeared** in a training corpus.

### Core Idea

A language model assigns a probability to a sequence of words:

\[
P(w_1, w_2, \dots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})
\]

Since conditioning on the full history is infeasible, the **Markov assumption** limits the context to the previous *n-1* words → this gives us **N-gram models**.

| Model | Context Used | Formula |
|-------|--------------|---------|
| Unigram | None | \( P(w_i) \) |
| Bigram | 1 previous word | \( P(w_i \mid w_{i-1}) \) |
| Trigram | 2 previous words | \( P(w_i \mid w_{i-2}, w_{i-1}) \) |

### Bi-gram Example

Corpus:
```
"I love NLP"
"I love coding"
"I hate bugs"
```

Counting pairs:

| Pair | Count |
|------|-------|
| (I, love) | 2 |
| (I, hate) | 1 |
| (love, NLP) | 1 |
| (love, coding) | 1 |

Probabilities:

\[
P(\text{love} \mid \text{I}) = \frac{2}{3}, \quad P(\text{hate} \mid \text{I}) = \frac{1}{3}
\]

So given `"I"`, the model predicts `"love"` with probability 0.67.

### Flaws of Statistical Models

- **Data Sparsity** — most valid word combinations never appear in the training corpus → zero probability.
- **No Generalization** — `"dog is running"` and `"puppy is running"` are treated as unrelated.
- **Short Context Window** — a trigram only remembers 2 previous words; long-range meaning is lost.
- **Curse of Dimensionality** — vocabulary size *V* with n-grams = *Vⁿ* combinations → explodes quickly.
- **No Semantic Understanding** — it counts surface forms, not meaning.
- **Smoothing Hacks Required** — Laplace, Kneser–Ney, etc., just to handle unseen n-grams.

> Stats-era models could *mimic* language patterns but never *understood* them.

---

## 2. Neural Era — RNNs & LSTMs

To fix sparsity and generalization issues, the field moved to **neural networks** that learn **dense vector representations** of words and model sequences continuously.

### Core Idea

- Represent each word as a **dense embedding** (a learned vector of real numbers).
- Feed these embeddings into a **Recurrent Neural Network (RNN)** that maintains a hidden state summarizing the sequence so far.
- Predict the next word from this hidden state.

```
x₁ → [RNN] → h₁
x₂ → [RNN] → h₂   (h₂ depends on h₁)
x₃ → [RNN] → h₃   (h₃ depends on h₂)
      ...
```

### Word2Vec — A Landmark Example

**Word2Vec** (Mikolov, 2013) learns word embeddings by predicting context words (CBOW / Skip-gram).

- Words with similar meanings end up close in vector space.
- Famous analogy: `king − man + woman ≈ queen`.
- First time models captured **semantic similarity** numerically.

### LSTM / GRU — Fixing Vanilla RNNs

Vanilla RNNs struggle with long sequences due to **vanishing gradients**.
**LSTMs (Long Short-Term Memory)** and **GRUs** add *gates* (input, forget, output) that decide what to remember vs. forget — enabling longer memory.

### Flaws of RNN/LSTM Models

- **Sequential Computation** — processes one token at a time → slow to train, hard to parallelize on GPUs.
- **Still Limited Long-Range Memory** — even LSTMs forget context beyond a few hundred tokens.
- **Information Bottleneck** — the whole input must be squeezed into a fixed-size hidden state.
- **Vanishing / Exploding Gradients** — mitigated but not solved.
- **Hard to Scale** — training on billions of tokens is impractical with recurrence.
- **Static Word Embeddings (Word2Vec era)** — the vector for `"bank"` is the same whether it's a riverbank or a bank account.

> Neural models understood *similarity* but still struggled with *context and scale*.

---

## 3. Transformer Era — The Rise of LLMs

Introduced in the 2017 paper *"Attention Is All You Need"*, the **Transformer** replaced recurrence entirely with a mechanism called **Self-Attention**, which lets every token look at every other token in the sequence simultaneously.

### LLM Definition

> **An LLM (Large Language Model) is a Transformer-based language model trained at massive scale (billions of parameters, trillions of tokens). Its core objective is still simple — predict the next token — but the architecture and scale make all the difference.**

Scale unlocks **emergent abilities**: reasoning, translation, coding, and in-context learning — none of which were explicitly programmed.

### How Transformers Solved the Previous Flaws

| Previous Flaw | Transformer Solution |
|---------------|---------------------|
| Short context (n-grams) | **Self-attention** → every token attends to every other token |
| Sequential bottleneck (RNNs) | **Parallel computation** → entire sequence processed at once |
| Static embeddings (Word2Vec) | **Contextual embeddings** → `"bank"` changes meaning based on sentence |
| Vanishing gradients | **Residual connections + LayerNorm** → stable deep training |
| Limited memory | **Attention over long sequences** (and now KV-caches, Flash Attention, long-context variants) |
| Data sparsity | **Pre-training on massive corpora** → model generalizes broadly |
| No word order in BoW | **Positional encodings** preserve token order |

### Types of LLMs

Transformers come in **three architectural flavors**, each suited for different tasks.

#### a) Encoder-Only — e.g. **BERT**

- Uses only the **encoder stack**.
- Reads the entire input **bidirectionally** (left + right context).
- Trained with **Masked Language Modeling (MLM)** — predict hidden words.
- **Best for *understanding* tasks**, not generation.

**Use cases:** classification, sentiment analysis, NER, semantic search, embeddings.

#### b) Decoder-Only — e.g. **GPT (Generative Pre-trained Transformer)**

- Uses only the **decoder stack** with **causal (masked) self-attention** — each token sees only previous tokens.
- Trained with **Next-Token Prediction**.
- Naturally suited for **generation**.

**Use cases:** chatbots, code generation, story writing, reasoning, agentic tasks.
**Examples:** GPT-4, Claude, LLaMA, Mistral, Gemini.

#### c) Encoder–Decoder (Seq2Seq) — e.g. **T5**, BART

- Encoder reads the input, decoder generates the output.
- Trained with **text-to-text** objectives (input text → output text).
- Excels at tasks where **input and output differ in structure**.

**Use cases:** translation, summarization, question answering, text-to-SQL.
**Examples:** T5, BART, mT5, FLAN-T5.

### When to Use Which Type of Model

| Task | Best Architecture | Why |
|------|------------------|-----|
| Text classification, sentiment, NER | **Encoder-Only (BERT)** | Needs full bidirectional understanding, no generation required |
| Semantic search / embeddings | **Encoder-Only** | Produces high-quality fixed representations |
| Chatbots, code gen, reasoning, open-ended generation | **Decoder-Only (GPT-style)** | Optimized for fluent next-token generation |
| Translation, summarization, Q&A | **Encoder-Decoder (T5, BART)** | Input is transformed into a distinct output sequence |
| Instruction-following assistants | **Decoder-Only + RLHF** | Current industry standard (ChatGPT, Claude, Gemini) |
| Multi-modal (image + text) | Decoder-Only with vision encoder | e.g. GPT-4V, Gemini, Claude 3 |

---

## Evolution Timeline

```
1990s ── N-grams / Markov Chains       (counting words)
2013 ── Word2Vec                       (dense embeddings)
2014 ── Seq2Seq (RNN Encoder-Decoder)  (sequence learning)
2015 ── Attention Mechanism            (focus on relevant tokens)
2017 ── Transformer ("Attention is all you need")
2018 ── BERT  (Encoder-Only)
2018 ── GPT-1 (Decoder-Only)
2019 ── T5    (Encoder-Decoder, text-to-text)
2020 ── GPT-3 (scale unlocks emergent abilities)
2022 ── ChatGPT (RLHF-tuned decoder-only LLM)
2023+ ── GPT-4, Claude, Gemini, LLaMA, multi-modal LLMs
```

---

## Key Takeaways

- **Stats Era** — probabilistic, memorized n-gram counts, no semantic understanding.
- **Neural Era** — learned dense embeddings and sequence models (RNN/LSTM), but sequential and short-memoried.
- **Transformer Era** — self-attention + massive scale → LLMs that generalize across tasks.
- **LLMs are still "just" next-token predictors** — the magic is in the **architecture (Transformers)** and **scale (data + parameters + compute)**.
- **Choose the architecture based on the task:**
  - Understand → **Encoder-Only (BERT)**
  - Generate → **Decoder-Only (GPT)**
  - Transform → **Encoder-Decoder (T5)**
