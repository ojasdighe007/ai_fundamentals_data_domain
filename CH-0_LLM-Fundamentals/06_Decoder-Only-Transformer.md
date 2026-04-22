# Decoder-Only Transformer

Decoder-only Transformers (GPT, LLaMA, Mistral, Claude, Gemini's generative core) are the workhorses of modern generative AI. They are trained with a single objective — **predict the next token given all previous tokens** — and use **causal (masked) self-attention** so that each position can only "see" the tokens to its left.

This note walks through two flows:

1. **Detailed internal flow** — how a single Transformer block turns static embeddings into *contextual* embeddings.
2. **High-level end-to-end flow** — from raw input text to the next-token prediction at the LM head.

---

## 1. High-Level Flow (Bird's-Eye View)

```
Input Text  ──►  Transformer Block Stack (× N)  ──►  LM Head  ──►  Next Token
```

### 1.1 Input Text

- Raw human-readable string (e.g., `"The cat sat on the"`).
- Needs to be converted into numbers before the neural network can process it.

### 1.2 Transformer Block Stack (× N)

- A **stack of N identical Transformer blocks** (e.g., 12 in GPT-2 small, 96 in GPT-3 175B, 80 in LLaMA-70B).
- Each block contains: **Masked Multi-Head Self-Attention → Feed-Forward Network**, wrapped with **Residual Connections + LayerNorm**.
- **Why stack them?** One block captures shallow relationships (syntax, local patterns); deeper blocks capture abstract semantics, reasoning, and long-range dependencies. Depth is a major driver of model capability.

### 1.3 LM Head (Language Modeling Head)

- A final **linear projection** that maps the last hidden state (dimension `d_model`) to a vector of size **`V` (vocabulary size)** — called **logits**.
- A **softmax** over logits produces a probability distribution over the entire vocabulary.
- The next token is then **sampled** (greedy, top-k, top-p, or temperature sampling).
- **Why needed?** The Transformer stack outputs abstract vectors; the LM head translates those vectors back into *token probabilities* — the only thing a language model ultimately emits.

> In most modern LLMs the LM head **shares weights with the input embedding matrix** (weight tying) — it saves parameters and improves generalization.

---

## 2. Detailed Internal Flow (Inside One Transformer Block)

```
Words ──► Tokens ──► Static Embeddings ──► + Positional Encoding
      ──► Q, K, V Vectors ──► Q·Kᵀ (relevancy scores) ──► Scale ──► Causal Mask
      ──► Softmax ──► × V (Value) ──► Delta Embeddings (context)
      ──► + Static Embeddings ──► Contextual Embeddings
```

Below is each step, what it does, and **why it is required**.

---

### 2.1 Words → Tokens

- The input string is split by a **tokenizer** (BPE, WordPiece, SentencePiece) into sub-word units and mapped to integer **token IDs**.
- Example: `"ChatGPT is great"` → `["Chat", "G", "PT", " is", " great"]` → `[7061, 38, 2898, 318, 1049]`.

**Why required?**

- Neural networks operate on numbers, not text.
- Sub-word tokenization keeps the **vocabulary small** (~30k–100k) while still covering any possible input — including rare words, typos, code, and multiple languages.

---

### 2.2 Tokens → Static Embeddings

- Each token ID indexes into a learned **embedding matrix** `E ∈ ℝ^(V × d_model)`.
- Output: a sequence of **dense vectors** (e.g., 768-dim in GPT-2, 4096-dim in LLaMA-7B).

**Why required?**

- A raw integer ID has no semantic meaning.
- Embeddings place tokens in a **continuous semantic space** where geometric proximity ≈ meaning similarity (`king − man + woman ≈ queen`).
- These vectors are **learnable parameters**, refined during training.

> These are called **static** at this stage because, before attention, every occurrence of the token `"bank"` has the same vector — regardless of context.

---

### 2.3 Static Embeddings + Positional Encoding

- A **positional vector** is added (or combined) with each token embedding to inject **order information**.
- Variants:
  - **Sinusoidal** (original Transformer)
  - **Learned absolute positions** (GPT-2)
  - **Rotary Position Embeddings (RoPE)** — used in LLaMA, Mistral, modern LLMs
  - **ALiBi** — used in some long-context models

**Why required?**

- Self-attention is **permutation-invariant** — without positional info, `"dog bites man"` and `"man bites dog"` would look identical to the model.
- Positional encodings give each token a unique "address" in the sequence so the model can distinguish order and relative distance.

---

### 2.4 Compute Query (Q), Key (K), Value (V) Vectors

For each token embedding `x` (a row vector of size `d_model`), the block computes three projections via learned weight matrices:

```
Q = x · W_Q          # "What am I looking for?"
K = x · W_K          # "What do I offer to others?"
V = x · W_V          # "What information do I carry?"
```

#### Analogy — A Library Search

Think of every token as a book that has:

| Symbol | Role            | Library analogy                                         |
| ------ | --------------- | ------------------------------------------------------- |
| **Q**  | Query vector    | The search query you type into the library catalog.    |
| **K**  | Key vector      | The book's **title / index card** — used for matching.  |
| **V**  | Value vector    | The actual **contents of the book** you'd read later.   |

The flow:

1. You submit your **query (Q)**.
2. The catalog compares it against every book's **key (K)** to decide *how relevant* each book is → this is the `Q · Kᵀ` step.
3. You then **pull information from the `V` (value) vectors** of the most relevant books.

> **Important:** the Value (`V`) is **not used here** — it is only stored at this stage and used later in step **2.9** (after masking + softmax produce attention weights). Think of Q/K as the "matching" stage and V as the "retrieval" stage.

#### Role of Each Vector in One Line

- **Q (Query)** → *asks*: "Which previous tokens are relevant to me right now?"
- **K (Key)** → *advertises*: "Here's a compact summary of what I can offer."
- **V (Value)** → *delivers*: "If you pick me, here's the actual information I'll hand over."

#### Why Three Separate Projections?

- If the model used the raw embedding for Q, K, and V, it would be forced to use *the same representation* for asking, advertising, and delivering — a severe limitation.
- Learning **three independent weight matrices** (`W_Q`, `W_K`, `W_V`) lets the model specialize each role. For example, a token might advertise itself as a *verb* (via K) but deliver *tense information* (via V) when attended to.

#### What is a "Head"? (explaining `dₖ`)

An **attention head** is one *independent parallel slice* of the attention mechanism. Instead of running one giant attention calculation over the full `d_model` dimension, the model splits it into `H` smaller attention calculations — each called a head.

```
d_model = 768       (e.g., GPT-2 small)
H       = 12        (number of heads)
d_k     = d_model / H = 64   ← dimension per head
```

Inside each head:

- `W_Q`, `W_K`, `W_V` project `x` down to a **`d_k`-dimensional subspace** (e.g., 64).
- The full attention computation (dot product → scale → mask → softmax → × V) runs *inside* that 64-dim subspace.
- After each head produces its own output, all `H` outputs are **concatenated** back into a `d_model` vector and passed through a final projection matrix `W_O`.

**Why multiple heads?**

- Different heads learn to focus on **different kinds of relationships** simultaneously:
  - One head might track **subject–verb agreement**.
  - Another might resolve **pronoun coreference** ("it" → "the cat").
  - Another might attend to **long-range topic** tokens far back in the sequence.
- It's like having **multiple specialists** in the same conversation, each listening for a different signal, and then combining their notes.

So when we write "scale by `√dₖ`", `dₖ` is the **dimension of a single head's key vector** (e.g., 64), *not* the full model dimension (768).

---

### 2.5 Dot Product Q · Kᵀ → Relevancy Score Matrix

- Compute `scores = Q · Kᵀ` → an **`n × n` matrix** (where `n` = sequence length).
- Entry `scores[i][j]` = how much token `i` should attend to token `j`.

**Why required?**

- The dot product is a natural **similarity measure** in vector space — high when Q and K point in similar directions.
- This matrix *is* the attention mechanism's "relevancy map": it decides which past tokens matter for predicting the current one.

---

### 2.6 Scale by √dₖ *(important step — often missed)*

```
scores = (Q · Kᵀ) / √dₖ
```

where `dₖ` is the dimension of each key vector (per head).

**Why required?**

- For large `dₖ`, raw dot products grow large in magnitude, pushing softmax into regions with **extremely small gradients** (saturation).
- Dividing by `√dₖ` keeps variance ~1, stabilizing training and preventing gradient collapse.

---

### 2.7 Causal Masking

- Apply a **lower-triangular mask**: set `scores[i][j] = −∞` for all `j > i`.
- Result: each token can only attend to itself and **tokens to its left** (past tokens).

**Why required?**

- This is the defining feature of a **decoder-only** model.
- During training, the model sees the entire sequence at once but must predict token `t+1` using **only tokens ≤ t** — otherwise it would trivially "cheat" by peeking at the answer.
- At inference, this guarantees autoregressive generation: tokens are produced one at a time, each conditioned only on what came before.

---

### 2.8 Softmax → Attention Weights

```
attention_weights = softmax(scores)
```

- Converts each row of the scores matrix into a **probability distribution** summing to 1.
- Masked positions (`−∞`) become 0.

**Why required?**

- Raw scores can be any real number; softmax turns them into **normalized weights** that can be interpreted as "attention percentages."
- This makes the subsequent weighted sum stable and bounded.

---

### 2.9 Multiply by Value Vectors → Delta Embeddings (Context)

```
context_i = Σ_j (attention_weights[i][j] × V_j)
```

- Each token's **context vector** is a weighted combination of all (allowed) Value vectors.
- This is what the user's note calls **"Delta Embeddings"** — they represent *what new contextual information each token has gathered from its neighbors*.

**Why required?**

- This is the step where **actual information flow** happens between tokens.
- The attention weights decided *who* to listen to; multiplying by V decides *what* to take from them.

---

### 2.10 Residual Connection: Delta + Static Embeddings → Contextual Embeddings

```
contextual_embedding = static_embedding + context (delta)
```

**Why required?**

- This is the **residual (skip) connection** around the attention block.
- Benefits:
  - **Preserves original information** — the model never "forgets" the token's own identity while mixing in context.
  - **Enables deep stacking** — gradients flow directly through the skip path, preventing vanishing gradients in deep networks (the same idea that made ResNets work).
  - Makes the block learn a **delta/update** rather than a full replacement, which is easier to optimize.

The output is then passed through **LayerNorm (or RMSNorm in LLaMA-family)** for stable activations.

---

### 2.11 Feed-Forward Network (FFN) — *missing from the original flow*

After attention, each token vector is passed independently through a **position-wise MLP**:

```
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂
```

- Typical activations: GELU (GPT-2), SwiGLU (LLaMA, Mistral), ReLU (original Transformer).
- Hidden dimension is usually **4× `d_model`** (e.g., 3072 for 768-dim models).
- Wrapped with another **residual connection + LayerNorm**.

**Why required?**

- Attention is essentially a **linear combination** of value vectors — it mixes tokens but doesn't add non-linear transformation power per token.
- The FFN injects **non-linearity** and increases the model's **capacity to store factual / world knowledge** — research has shown that FFN layers act as key-value memories for facts.

---

### 2.12 Repeat for N Blocks

The output of block `k` becomes the input of block `k+1`. After all `N` blocks, we get the **final hidden states** — one contextual vector per input token.

**Why required?**

- Each block refines the representation further:
  - Lower layers → token-level / syntactic features.
  - Middle layers → phrase and clause-level semantics.
  - Upper layers → abstract meaning, task-level reasoning.

---

### 2.13 LM Head + Sampling

Taking the **final hidden state of the last token** only (at inference):

```
logits = final_hidden · Wᵀ_E        # Wᵀ_E is often tied to the input embedding matrix
probs  = softmax(logits)
next_token = sample(probs)          # greedy / top-k / top-p / temperature
```

**Why required?**

- `logits`: raw scores per vocabulary entry.
- `softmax`: converts logits to a valid probability distribution.
- **Sampling strategy** controls creativity vs. determinism:
  - **Greedy** — always pick the max → deterministic but repetitive.
  - **Temperature** — scales logits; high = more random, low = more focused.
  - **Top-k / Top-p (nucleus)** — restrict sampling to the most probable tokens to avoid incoherent outputs.

The chosen token is appended to the input, and the whole process repeats **autoregressively** until an end-of-sequence token or max length is reached.

---

## 3. Deep Dive — Core Components Explained with Analogies

Three components appear again and again in every Transformer block. Here's what each one *means* in plain language.

---

### 3.1 Masked Multi-Head Self-Attention (MMHSA)

Let's break the name apart:

| Part              | Meaning                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| **Self**          | Every token attends to **other tokens in the same sequence** (not a different sentence).         |
| **Attention**     | A weighted lookup: pick who to listen to, then combine their information.                        |
| **Multi-Head**    | Run `H` independent attention operations in parallel, each in its own subspace.                  |
| **Masked**        | Block future tokens so the model can only use **past and current** tokens (causal generation).   |

#### Generic Analogy — A Classroom Discussion

Imagine a classroom where students take turns speaking. When it's **Student 5's** turn:

- **Self-attention:** Student 5 listens to everyone else in the room (same classroom, not another school).
- **Masked:** Students 6, 7, 8… haven't spoken yet, so Student 5 is **blindfolded** to their notes — only Students 1–5's notes are visible.
- **Multi-Head:** Student 5 has **12 pairs of ears**. Each pair listens for a different thing — one for topic, one for grammar cues, one for emotional tone, one for references to earlier names, etc.
- **Attention weights:** For each pair of ears, Student 5 decides how much weight to give each earlier student ("Student 2 said something really relevant, I'll weight them 60%; Student 3 only 10%…").
- **Output:** Student 5 combines what they heard across all 12 ears into a single updated understanding and writes it in their notebook.

That updated notebook entry is the **contextual embedding** for Student 5 after this block.

---

### 3.2 Feed-Forward Network (FFN)

After attention mixes information *between* tokens, the FFN processes **each token independently** through a small 2-layer neural network:

```
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂
```

- `W₁` expands the vector (e.g., 768 → 3072).
- A non-linear activation (GELU / SwiGLU / ReLU) is applied.
- `W₂` projects it back (3072 → 768).

#### Generic Analogy — Individual Reflection After a Group Meeting

- **Attention = the meeting:** everyone shares ideas and you absorb them.
- **FFN = going back to your desk alone to think it over**, transform the raw notes into your own structured understanding, and update your mental model.

Why the *expand-then-contract* shape?

- The **wide hidden layer (4× d_model)** gives the network room to combine features in complex, non-linear ways.
- The non-linear activation (e.g., GELU) is what allows the FFN to learn **non-linear functions** — plain matrix multiplications alone can only learn linear mappings.
- Recent research shows FFN layers behave like **key-value memories** for factual knowledge (e.g., "Paris is the capital of France" is stored here, not in attention).

> **Rule of thumb:** Attention handles *communication between tokens*; FFN handles *computation within a token*.

---

### 3.3 Residual Connections + LayerNorm

These two always appear together, wrapping both the attention block and the FFN block.

#### Residual (Skip) Connection

```
output = x + Sublayer(x)          # Sublayer = attention or FFN
```

The input `x` is **added back** to the sublayer's output.

##### Generic Analogy — Editing a Document with "Track Changes"

- Without a residual: each editor **rewrites the whole document** from scratch → errors compound, original intent gets lost.
- With a residual: each editor **suggests changes on top of the original** → you always keep the original text and just layer edits on top.

**Why it matters:**

- **Preserves the original signal** — the model can always "fall back" on the input if the sublayer doesn't help.
- **Gradients flow straight through** the skip path during backpropagation → fixes the **vanishing gradient problem** and lets us stack 50–100+ layers (same trick behind ResNet in computer vision).

#### LayerNorm (Layer Normalization)

For each token vector `x` independently:

```
LayerNorm(x) = γ · (x − mean(x)) / std(x) + β
```

- Subtract the mean and divide by the standard deviation across the vector's features → **zero mean, unit variance**.
- Then scale by learned `γ` and shift by learned `β` so the network can still re-introduce any needed offset.

##### Generic Analogy — Normalizing Volume Levels

- Imagine each token vector is an **audio track**. Some tracks are very loud (large activations), some very quiet (small activations).
- **LayerNorm = an audio leveler** — it brings every track to roughly the same volume so downstream layers don't get "deafened" by loud tokens or miss the quiet ones.
- `γ` and `β` are like **volume knobs** the model can adjust during training if it *wants* a particular track louder or softer.

> **Modern note:** LLaMA-family models use **RMSNorm** (a simpler variant without mean subtraction) which is faster and works equally well in practice.

#### Pre-Norm vs. Post-Norm

- **Original Transformer (2017):** `x + Sublayer(LayerNorm later)` — post-norm.
- **Modern LLMs (GPT-2, LLaMA, etc.):** `x + Sublayer(LayerNorm(x))` — **pre-norm**, which is more stable for deep training.

---

## 4. What Gets Trained? (Learnable Parameters)

When we say "training an LLM," we are learning the values inside a specific set of matrices and vectors. Everything else (the softmax, dot products, masking, etc.) is **fixed math** with no parameters.

### 4.1 Inventory of Trainable Components

| Component                         | Shape (typical)                  | Role                                                      |
| --------------------------------- | -------------------------------- | --------------------------------------------------------- |
| **Token embedding matrix** `E`    | `V × d_model`                    | Turns token IDs → vectors.                                |
| **Positional embeddings**         | `L × d_model` (if learned)       | Encode token order. *Fixed* for sinusoidal; *learned* for GPT-2; *rotary* (RoPE) has no extra params. |
| **Attention `W_Q` (per head)**    | `d_model × d_k`                  | Projects input to the Query subspace.                     |
| **Attention `W_K` (per head)**    | `d_model × d_k`                  | Projects input to the Key subspace.                       |
| **Attention `W_V` (per head)**    | `d_model × d_v`                  | Projects input to the Value subspace.                     |
| **Output projection `W_O`**       | `d_model × d_model`              | Combines all heads back into one vector.                  |
| **FFN weights** `W₁`, `b₁`        | `d_model × 4·d_model`, `4·d_model` | First layer of the FFN (expand).                        |
| **FFN weights** `W₂`, `b₂`        | `4·d_model × d_model`, `d_model` | Second layer of the FFN (project back).                   |
| **LayerNorm params** `γ`, `β`     | `d_model`, `d_model`             | Scale + shift after normalization.                        |
| **LM head** `W_LM`                | `d_model × V` (often tied to `E`) | Projects final hidden state to vocabulary logits.        |

All of the above are **duplicated per Transformer block** — so a 96-layer model has 96 copies of the attention and FFN weights (each with its own learned values).

### 4.2 What is NOT Trained

- **The dot product** `Q · Kᵀ` — pure math.
- **The scaling factor** `√dₖ` — a constant.
- **The causal mask** — a fixed triangular matrix of 0s and −∞.
- **The softmax function** — parameter-free.
- **Positional encodings if sinusoidal or RoPE** — deterministic functions of position.

### 4.3 How Training Works (One Line)

1. **Forward pass:** input text → predicted probability distribution for the next token.
2. **Loss:** **cross-entropy** between predicted distribution and the actual next token.
3. **Backward pass:** compute gradients of the loss w.r.t. *every trainable parameter above*.
4. **Optimizer step:** update each parameter (usually with **AdamW**) to reduce the loss.
5. Repeat over trillions of tokens.

> **Scale example:** GPT-3 has **175 billion** trainable parameters, distributed across 96 Transformer blocks. Each parameter is a single `float` number that gets nudged during training.

---

## 5. Putting It All Together — Block Diagram

```
                  ┌──────────────────────────────────────────┐
                  │              Transformer Block            │
                  │                                          │
Input Embeddings ─┼─► LayerNorm ─► Masked Multi-Head Attn ──┐│
        │         │                                          ▼│
        └─────────┼──────────────────── + (residual) ────────┤│  ← contextual embeddings
                  │                                          ││
                  │  LayerNorm ─► Feed-Forward Network ──┐   ││
                  │                                       ▼   ││
                  │  ───────────── + (residual) ──────────┘   │
                  └──────────────────────────────────────────┘
                                    │
                              (repeat × N)
                                    │
                                    ▼
                           Final LayerNorm
                                    │
                                    ▼
                                 LM Head  (linear → logits)
                                    │
                                    ▼
                             Softmax + Sampling
                                    │
                                    ▼
                               Next Token
```

---

## 6. Corrections & Enhancements to the Original Flow

| # | Original Step | Issue / Enhancement |
|---|---------------|---------------------|
| 1 | `Query → Key → Dot Product` | Q, K, V are all computed **in parallel** from the same input — not sequentially. |
| 2 | Missing | **Scaling by √dₖ** before softmax — critical for stable gradients. |
| 3 | Missing | **Multi-Head Attention** — multiple attention heads run in parallel and are concatenated. |
| 4 | Masking → Softmax → Value | Order is correct, but masking happens *inside* the scaled-score matrix **before** softmax (set to −∞), not as a separate post-softmax step. |
| 5 | Missing | **Feed-Forward Network (FFN)** after attention — adds non-linearity and stores knowledge. |
| 6 | Missing | **Residual connections + LayerNorm** (pre-norm in modern LLMs) around both attention and FFN. |
| 7 | "Sum Delta + Static = Contextual" | Correct intuition — this *is* the residual connection; worth calling it out explicitly. |
| 8 | Missing (high-level) | **Sampling strategy** after the LM head — the mechanism that actually turns probabilities into a generated token. |

---

## 7. Key Takeaways

- Decoder-only Transformers are **causal, autoregressive next-token predictors**.
- Attention turns **static embeddings → contextual embeddings** by letting each token pull information from previous tokens via **Q·K similarity + V weighted sum**.
- **Masking** is what makes the model "decoder-only" — it enforces left-to-right prediction.
- **Residual connections + LayerNorm + FFN** are not optional — they are what make deep stacks trainable and expressive.
- The **LM head + softmax + sampling** is the bridge from hidden vectors back to actual generated tokens.
- Scale (depth, width, heads, training data) is what transforms this architecture into a *Large* Language Model with emergent abilities.
