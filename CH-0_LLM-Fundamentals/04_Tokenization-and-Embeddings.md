# Tokenization and Embeddings

Before a language model can "understand" text, it has to convert **human-readable strings** into **numbers** the neural network can process. This happens in two stages:

```
Raw Text ──► Tokenization ──► Token IDs ──► Embeddings ──► Model
 "I love AI"   ["I","love","AI"]   [40, 1842, 15836]   [vectors]
```

---

## 1. What is Tokenization and Token ID?

### Tokenization

**Tokenization** is the process of splitting raw text into smaller units called **tokens**. A token can be a **word**, **sub-word**, or **character**, depending on the tokenizer.

Example (OpenAI's `cl100k_base` tokenizer):

```
"I love tokenization!"  →  ["I", " love", " token", "ization", "!"]
```

Notice that `"tokenization"` is split into two sub-word tokens — this is intentional and very common in modern LLMs.

### Token ID

Each token in a tokenizer's **vocabulary** is assigned a **unique integer ID**. The model never sees text — it only sees these IDs.

```
"I"         → 40
" love"     → 1842
" token"    → 4037
"ization"   → 2065
"!"         → 0
```

So the sentence becomes `[40, 1842, 4037, 2065, 0]` before being passed to the model.

### Types of Tokenization


| Type                         | How it works                                | Example (`"unhappiness"`)                       | Used by              |
| ---------------------------- | ------------------------------------------- | ----------------------------------------------- | -------------------- |
| **Word-level**               | Split on whitespace/punctuation             | `["unhappiness"]`                               | Early NLP, Word2Vec  |
| **Character-level**          | Every character is a token                  | `["u","n","h","a","p","p","i","n","e","s","s"]` | Niche / multilingual |
| **Byte-Pair Encoding (BPE)** | Merges frequent byte pairs into sub-words   | `["un", "happiness"]`                           | GPT-2, GPT-3, GPT-4  |
| **WordPiece**                | Variant of BPE with likelihood-based merges | `["un", "##happiness"]`                         | BERT                 |
| **SentencePiece / Unigram**  | Language-agnostic, works on raw bytes       | `["▁un", "happiness"]`                          | T5, LLaMA, Gemini    |


### Why Sub-word Tokenization Wins

- Handles **unknown / rare words** by decomposing them (`"GPT-42"` → `["GPT", "-", "42"]`).
- Keeps **vocabulary small** (~30k–100k tokens) while covering any text.
- Works across **multiple languages** and special characters.
- Reduces the **out-of-vocabulary (OOV)** problem that plagued older models.

### Rule of Thumb

> 1 token ≈ **4 characters** of English text ≈ **0.75 words**.
> 1,000 tokens ≈ **750 words** ≈ a short page.

---

## 2. Embeddings

A **token ID** is just an integer — it carries no meaning. An **embedding** is a **dense vector of real numbers** (typically 256–12,288 dimensions) that represents that token in a **semantic space**, where similar tokens end up close together.

```
Token ID 1842 ("love")  →  [0.21, -0.87, 0.44, ..., 0.13]   (e.g. 1536 dims)
```

Embeddings are **learned during training** — the model adjusts them so that words used in similar contexts get similar vectors.

Classic analogy (Word2Vec):

```
king  − man  + woman  ≈  queen
Paris − France + Italy ≈  Rome
```

---

### Types of Embeddings

There are **three main generations** of embeddings:

#### a) Static / Classical Embeddings

Each word has **one fixed vector**, regardless of context.


| Model        | Idea                                                                      |
| ------------ | ------------------------------------------------------------------------- |
| **Word2Vec** | Predict context words from a target word (Skip-gram) or vice versa (CBOW) |
| **GloVe**    | Factorize a global word co-occurrence matrix                              |
| **FastText** | Word2Vec + sub-word n-grams → handles rare words                          |


**Limitation:** the vector for `"bank"` is the same in `"river bank"` and `"bank account"`.

#### b) Contextual Embeddings

The vector for a token **changes based on the surrounding sentence**.


| Model                | Architecture             | What's special                          |
| -------------------- | ------------------------ | --------------------------------------- |
| **ELMo**             | Bi-LSTM                  | First popular contextual embedding      |
| **BERT**             | Encoder-only Transformer | Bidirectional context, MLM pre-training |
| **RoBERTa, DeBERTa** | BERT variants            | Better training recipes                 |
| **GPT embeddings**   | Decoder-only Transformer | Causal, used inside generative models   |


**Example:**

```
"I deposited money at the bank"     →  bank ≈ [finance-ish vector]
"We had a picnic on the river bank" →  bank ≈ [geography-ish vector]
```

#### c) Sentence / Document Embeddings

Represent an **entire sentence or passage** as a single vector — essential for **semantic search, RAG, clustering**.


| Model                                        | Use case                                                   |
| -------------------------------------------- | ---------------------------------------------------------- |
| **Sentence-BERT (SBERT)**                    | Semantic similarity, fine-tuned BERT with siamese networks |
| **OpenAI `text-embedding-3-small / -large`** | General-purpose retrieval                                  |
| **Cohere Embed, Voyage AI, BGE, E5**         | RAG-optimized open/closed embeddings                       |


---

### How Embeddings Are Calculated (Concise Example)

#### Word2Vec (Skip-gram) — training intuition

Goal: given a **center word**, predict its **context words** within a window.

Corpus: `"I love natural language processing"`, window = 2

Training pairs (center → context):

```
(love, I),  (love, natural),  (love, language)
(natural, love),  (natural, language),  (natural, processing)
```

A shallow neural network learns a weight matrix `W ∈ R^{V × d}` (V = vocab size, d = embedding dim). After training, **row *i* of W is the embedding for word *i***. Words appearing in similar contexts end up with similar rows.

#### Transformer-based Embeddings — inference intuition

1. Tokenize text → token IDs.
2. **Look up** each token's vector from the model's **embedding matrix** (these are learned weights).
3. Add **positional encodings** so the model knows token order.
4. Pass through **N Transformer layers** — self-attention updates each token's vector using context.
5. The **final layer's hidden states** are the **contextual embeddings**.
6. For calculating a **sentence embedding using these contextual embeddings**, pool the token vectors (mean-pool, CLS token, or last-token pool).

```
Tokens      →  Embedding Matrix Lookup  →  + Positional Encoding
            →  Self-Attention × N layers →  Pool  →  Sentence Embedding
```

---

### Which Embeddings Use Which Kind of Transformer?


| Embedding Type                   | Transformer Architecture                             | Why it's suited                                                     |
| -------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| **Word2Vec / GloVe / FastText**  | ❌ No Transformer (shallow NN / matrix factorization) | Pre-Transformer era                                                 |
| **BERT / RoBERTa / DeBERTa**     | **Encoder-Only**                                     | Bidirectional context → best for understanding                      |
| **Sentence-BERT, BGE, E5, GTE**  | **Encoder-Only (fine-tuned)**                        | Optimized for sentence similarity / retrieval                       |
| **T5 / FLAN-T5 encoder outputs** | **Encoder–Decoder** (encoder side)                   | Useful when task is text-to-text                                    |
| **GPT / LLaMA embeddings**       | **Decoder-Only**                                     | Causal; used internally for generation, can be pooled for retrieval |
| **OpenAI `text-embedding-3-*`**  | Decoder-Only based (pooled)                          | General-purpose retrieval at scale                                  |


> **Rule of thumb:** for **retrieval / semantic search / RAG**, prefer **encoder-only** or encoder-pooled embeddings (BERT-family, SBERT, BGE, OpenAI embeddings). For **generation**, decoder-only models use their internal embeddings implicitly.

---

## 3. The End-to-End Flow

> **Each word → token → token ID → embedding vector. This mapping changes from model to model — every LLM has its *own* tokenizer and its *own* embedding matrix.**

```
"I love AI"
   │
   ▼  Tokenizer (model-specific)
["I", " love", " AI"]
   │
   ▼  Vocabulary lookup
[40, 1842, 15836]          ← Token IDs
   │
   ▼  Embedding matrix lookup (learned weights)
[[0.21, -0.87, ...],        ← Embedding vectors
 [0.04,  0.55, ...],
 [-0.12, 0.33, ...]]
   │
   ▼  + Positional encoding
   ▼  Transformer layers (self-attention)
Contextual embeddings / model output
```

### Why the Mapping Differs Between Models

- Different tokenizers → **different vocabularies** and IDs.
*Example:* the word `"ChatGPT"` might be **1 token** in GPT-4 but **3 tokens** in LLaMA.
- Different training data and objectives → **different embedding matrices**.
- Different dimensions (e.g. 768 for BERT-base, 1536 for `text-embedding-3-small`, 4096 for LLaMA-7B).
- **You cannot mix** embeddings across models — a vector from OpenAI's embedding model is not comparable to a vector from BERT.

---

## Key Takeaways

- **Tokenization** splits text into tokens; each token gets a **unique integer ID** from the model's vocabulary.
- Modern LLMs use **sub-word tokenization** (BPE, WordPiece, SentencePiece) to balance vocabulary size and coverage.
- **Embeddings** turn token IDs into **dense semantic vectors** — words with similar meaning land in similar regions of space.
- Embeddings evolved from **static (Word2Vec/GloVe)** → **contextual (BERT/GPT)** → **sentence-level (SBERT, OpenAI embeddings)**.
- **Encoder-only Transformers** dominate retrieval/embedding tasks; **decoder-only** models use embeddings internally for generation.
- **Every model has its own tokenizer + embedding matrix** — embeddings are **not portable** across models.

