# Natural Language Processing (NLP)

## What is NLP?

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence that enables machines to **read, understand, interpret, and generate human language** (text or speech) in a way that is both meaningful and useful.

It sits at the intersection of:
- **Linguistics** — the rules and structure of language
- **Computer Science** — algorithms and data structures
- **Machine Learning / Deep Learning** — pattern recognition from data

> In short: NLP bridges the gap between how humans communicate and how computers process information.

---

## Why is NLP Hard?

Human language is inherently:
- **Ambiguous** — "I saw the man with the telescope" (who has the telescope?)
- **Context-dependent** — "bank" (river bank vs. financial bank)
- **Unstructured** — free-form text, slang, typos, sarcasm
- **Multilingual** — thousands of languages, dialects, and scripts
- **Evolving** — new words, memes, and meanings appear constantly

---

## Core Goals of NLP

1. **Understanding (NLU)** — extract meaning from text (intent, entities, sentiment)
2. **Generation (NLG)** — produce fluent, coherent text (summaries, answers, translations)
3. **Interaction** — enable conversation between humans and machines

---

## Common NLP Techniques

NLP techniques are typically applied as a **pipeline**, from raw text → cleaned tokens → features → model output.

### 1. Text Preprocessing

| Technique | Purpose | Example |
|-----------|---------|---------|
| **Tokenization** | Split text into words/subwords/sentences | `"I love NLP"` → `["I", "love", "NLP"]` |
| **Lowercasing** | Normalize case | `"Apple"` → `"apple"` |
| **Stop-word Removal** | Drop common words with low info | Remove `the`, `is`, `a` |
| **Stemming** | Chop suffixes to root form (rule-based) | `running` → `run` |
| **Lemmatization** | Reduce to dictionary root (context-aware) | `better` → `good` |
| **Punctuation / Noise Removal** | Strip symbols, HTML, emojis | `"Hi!!!"` → `"Hi"` |

### 2. Syntactic Analysis (Structure)

- **Part-of-Speech (POS) Tagging** — label each word as noun, verb, adjective, etc.
- **Parsing (Dependency / Constituency)** — uncover grammatical structure and word relationships
- **Chunking** — group words into phrases (noun phrases, verb phrases)

### 3. Semantic Analysis (Meaning)

- **Named Entity Recognition (NER)** — identify people, places, organizations, dates
  - *Example:* `"Apple was founded in Cupertino"` → `Apple [ORG]`, `Cupertino [LOC]`
- **Word Sense Disambiguation (WSD)** — choose correct meaning based on context
- **Coreference Resolution** — link pronouns to entities (`"John said he..."` → `he = John`)
- **Semantic Role Labeling** — identify who did what to whom

### 4. Text Representation (Turning Words into Numbers)

| Technique | Description |
|-----------|-------------|
| **Bag-of-Words (BoW)** | Count of each word in a document (ignores order) |
| **TF-IDF** | Weighs words by how unique they are across documents |
| **One-Hot Encoding** | Binary vector per word (sparse) |
| **Word Embeddings** (Word2Vec, GloVe, FastText) | Dense vectors capturing semantic similarity |
| **Contextual Embeddings** (BERT, GPT) | Vectors that change based on surrounding context |

### 5. Core NLP Tasks

- **Text Classification** — spam detection, topic labeling
- **Sentiment Analysis** — positive / negative / neutral
- **Machine Translation** — English ↔ French, etc.
- **Summarization** — extractive or abstractive
- **Question Answering (QA)** — answer questions from context or knowledge
- **Text Generation** — stories, code, chat responses
- **Speech-to-Text / Text-to-Speech** — bridge spoken and written language
- **Information Retrieval** — search engines, semantic search

### 6. Modern Techniques (Deep Learning Era)

- **Recurrent Neural Networks (RNN / LSTM / GRU)** — sequential models for text
- **Attention Mechanism** — focus on relevant parts of input
- **Transformers** — parallelized architecture powering modern NLP (BERT, GPT, T5)
- **Pre-training + Fine-tuning** — train on massive corpora, adapt to specific tasks
- **Large Language Models (LLMs)** — general-purpose models like GPT-4, Claude, LLaMA

---

## Typical NLP Pipeline

```
Raw Text
   │
   ▼
Preprocessing  (clean, tokenize, normalize)
   │
   ▼
Feature Extraction  (BoW / TF-IDF / Embeddings)
   │
   ▼
Modeling  (ML / Deep Learning / LLM)
   │
   ▼
Task Output  (label, answer, translation, summary…)
```

---

## Real-World Applications

- Chatbots & virtual assistants (Siri, Alexa)
- Search engines (Google, semantic search)
- Email spam filters
- Grammar and autocorrect tools
- Language translation (Google Translate, DeepL)
- Content moderation on social media
- Resume screening and document intelligence
- Voice-controlled systems

---

## Key Takeaways

- NLP = teaching machines to **understand and generate** human language.
- It combines **linguistics, statistics, and deep learning**.
- Techniques span a **pipeline**: preprocessing → representation → modeling → task output.
- Modern NLP is dominated by **Transformer-based LLMs**, which power tools like ChatGPT, Claude, and Gemini.
