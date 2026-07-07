# NLP Frameworks — What Exists and What to Use

A practical map of the Python NLP ecosystem. The goal is not to pick *one* winner but to know **which tool does which job**, since real pipelines combine several of them.

> **Mental model:** don't ask *"which framework do I choose?"* — ask *"which task am I doing right now?"* A typical pipeline uses spaCy to clean text, scikit-learn / gensim / Hugging Face to vectorize, and scikit-learn again to train a model. They are teammates, not rivals.

---

## The landscape by purpose

### 1. Learning fundamentals
| Library | Purpose | Status |
|---|---|---|
| **NLTK** | Teaching toolkit. Exposes tons of classic algorithms (stemming, tokenization, parsing). Great for *understanding* how things work. | Still used in courses, but **slow and rarely used in production** today. |

### 2. Text processing (tokenizing, POS, NER, lemmatization)
| Library | Purpose | Status |
|---|---|---|
| **spaCy** | The production workhorse. Fast, opinionated, one clean way to do things. Tokenizing, POS tagging, NER, dependency parsing. | **Core / very common.** |
| **Stanza** (Stanford) | spaCy-like pipelines with strong **multilingual** accuracy. | Common in multilingual work. |
| **CoreNLP** (Stanford) | Older Java-based linguistic toolkit. | **Largely legacy** — you'll see it cited, rarely started fresh. |

### 3. Vectorization — turning text into numbers
| Library | Purpose | Status |
|---|---|---|
| **scikit-learn** | *Sparse* vectors: `CountVectorizer`, `TF-IDF`. Plus general ML (classifiers, clustering). The bridge from text-as-numbers to models. | **Core / essential.** |
| **gensim** | *Classic dense* embeddings & topic modeling: Word2Vec, Doc2Vec, FastText, LDA. Purpose-built for unsupervised semantic vectors without a heavy deep-learning stack. | Still used, but **classic Word2Vec-style embeddings are less central nowadays** — mostly topic modeling and lightweight/offline cases. |
| **fastText** (Meta) | Word2Vec's cousin: word embeddings + very fast text classification, handles subwords. | Niche but alive for fast classification. |
| **Hugging Face `transformers`** | *Modern contextual* dense vectors: BERT, GPT-style models, embeddings. Where the field actually is today. | **Core / dominant.** |
| **`sentence-transformers`** | Sentence/document-level embeddings on top of transformers. The go-to for semantic similarity & retrieval. | **Core / dominant.** |

### 4. Deep-learning foundations
| Library | Purpose | Status |
|---|---|---|
| **PyTorch** | The underlying DL framework most NLP research & models run on. | **Core** (the default today). |
| **TensorFlow / Keras** | Alternative DL framework. | Still around, **losing ground to PyTorch** in NLP research. |
| **Flair** | Easy contextual embeddings + strong NER, on PyTorch. | Niche but handy. |
| **AllenNLP** | Research NLP on PyTorch. | **Mostly inactive** — you'll see it in older papers. |

### 5. Tokenization as its own layer (for modern models)
| Library | Purpose | Status |
|---|---|---|
| **Hugging Face `tokenizers`** | Fast subword tokenization for transformer models. | Core under the hood. |
| **SentencePiece** | Subword tokenization (used by many LLMs). | Common under the hood. |

### 6. Storing & searching embeddings (once you *have* vectors)
| Library | Purpose | Status |
|---|---|---|
| **FAISS**, **Annoy**, **hnswlib** | Fast similarity search over embedding vectors. | Common (FAISS is the classic). |
| **Chroma, Qdrant, Weaviate, Milvus, Pinecone** | Vector databases — the backbone of retrieval / RAG systems. | **Very common nowadays.** |

### 7. LLM app orchestration (newest wave)
| Library | Purpose | Status |
|---|---|---|
| **LangChain**, **LlamaIndex** | Build apps *on top of* LLMs: retrieval, chaining, agents, RAG. | **Very common nowadays.** |

### 8. Supporting tools (unglamorous but constant)
| Library | Purpose |
|---|---|
| **Hugging Face `datasets`** / **`evaluate`** | Loading benchmark datasets & computing metrics (BLEU, ROUGE, etc.). |
| **Data labeling**: Prodigy, Doccano, Label Studio | Creating training/annotation data. |
| **Cleaning**: ftfy, clean-text, regex | Fixing broken Unicode & messy text. |
| **scispaCy** | Biomedical/scientific domain spaCy. |

---

## The core set (what to actually learn first)

If you strip away the niche and legacy tools, the modern working core is:

- **scikit-learn** — sparse vectorization + classic ML models
- **spaCy** — text processing / cleaning
- **Hugging Face `transformers` + `sentence-transformers`** — modern embeddings & models
- **PyTorch** — the DL foundation underneath
- **A vector store (FAISS or a vector DB)** — once you work with embeddings
- **LangChain / LlamaIndex** — only if you're building LLM apps

## Most common *nowadays*
**Hugging Face (transformers / sentence-transformers), spaCy, scikit-learn, PyTorch**, and — for LLM/RAG apps — **vector databases + LangChain/LlamaIndex**.

## Fading / less used nowadays (still worth recognizing)
- **NLTK** — great for learning, rarely production.
- **CoreNLP** — legacy Java toolkit.
- **AllenNLP** — mostly inactive.
- **TensorFlow/Keras** — losing ground to PyTorch in NLP.
- **gensim / classic Word2Vec** — still fine for topic modeling & lightweight cases, but contextual embeddings have largely replaced classic static ones for most tasks.

---

## Core libraries for a multi-class text classification task

Two realistic routes depending on how much power you need.

### Route A — Classic & lightweight (fast, interpretable, small data)
**scikit-learn** (+ optionally spaCy for preprocessing).

```
spaCy / scikit-learn  →  TF-IDF  →  LogisticRegression / LinearSVC
```

- `spaCy` or scikit-learn's built-ins for tokenizing/cleaning
- `TfidfVectorizer` for features
- `LogisticRegression`, `LinearSVC`, or `RandomForest` as the classifier
- `train_test_split`, `classification_report` for evaluation

Best when: data is modest, you want speed, interpretability, and no GPU.

### Route B — Modern & state-of-the-art (best accuracy)
**Hugging Face `transformers`** on top of **PyTorch**, with **`datasets`** + **`evaluate`**.

```
Hugging Face tokenizer  →  fine-tune BERT/DistilBERT  →  Trainer  →  evaluate
```

- `transformers` — a pretrained model (e.g. `bert-base-uncased`, `distilbert`) fine-tuned for classification
- `datasets` — load/prepare your data
- `evaluate` — accuracy / F1
- `PyTorch` — the backend

Best when: you want top accuracy and have enough data (and ideally a GPU).

> **Note on multi-label:** if each text can have *several* labels at once (not just one), it's *multi-label* classification. Same tools — in Route A use `OneVsRestClassifier`; in Route B set the model's problem type to multi-label (sigmoid + BCE loss) instead of single-label softmax.

### Quick recommendation
- Prototyping or limited data → **Route A (scikit-learn + TF-IDF)**.
- Production or best accuracy → **Route B (Hugging Face transformers)**.