# BANKING77 — Intent Classification

check the work on lab.ipynb

## 📊 Dataset & Task Description

This project uses the **BANKING77** dataset (originally curated by PolyAI) to build and evaluate an intent classification system for financial customer support. The goal is not a single model but a **systematic comparison of text-representation strategies** — from classical sparse features through static and contextual dense embeddings to a fine-tuned transformer — all evaluated under the same protocol.

🔗 **Dataset:** [PolyAI/banking77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)

### 👥 Dataset Overview

BANKING77 is a specialized, fine-grained text classification benchmark comprising real-world customer service queries from the banking domain.

* **Total Samples:** 13,083 customer utterances.
* **Train/Test Split:** 10,003 training samples and 3,080 evaluation samples (official split, used as-is).
* **Granularity:** 77 distinct, fine-grained intent classes.
* **Test-set balance:** the test set is *perfectly balanced* (40 examples per class), while the training set is *mildly imbalanced* (~35–187 examples per class, ~5.3× ratio). This makes the balanced test set an ideal instrument for detecting whether the training imbalance harms rare classes.

### 🎯 The Machine Learning Task

The core task is **Single-Label, Multi-Class Text Classification**. Given a raw text input from a user, the model must predict exactly one correct intent out of the 77 possible categories.

#### Key Technical Challenges

* **Fine-Grained Semantic Overlap:** Many categories share almost identical vocabulary and differ only by subtle context, requiring high semantic precision.
* **Real-World Noise:** Inputs contain natural human variation — casual phrasing, typos, shorthand, and varying sentence lengths.
* **High-Dimensional Class Space:** A 77-way split raises misclassification risk relative to simpler binary or low-class tasks.
* **Keyword-Driven Signal:** Intent is often carried by a few decisive terms (`top up`, `PIN`, `IBAN`, `direct debit`), which strongly favours methods that preserve exact vocabulary.

## 🛠️ Data Engineering Pipeline (Polars)

Local Parquet files are loaded and manipulated with the **Polars** library for fast, memory-efficient handling before being passed into the modelling and tokenization pipelines. Polars is used for loading, class-support inspection, and imbalance analysis; feature extraction and modelling are handled by scikit-learn, PyTorch, and Hugging Face.

## 🧪 Methodology

All approaches share the same skeleton — **turn each query into a fixed-length vector, then classify** — and differ only in *how* that vector is built. Every model is evaluated with **macro-F1** as the primary metric (all 77 classes weighted equally, so rare classes count fully) alongside accuracy and micro/weighted variants. Hyperparameters are tuned via `GridSearchCV` with 5-fold cross-validation on the training set; the test set is touched only once, at the end.

### Representation strategies explored

1. **Sparse lexical (Bag-of-Words & TF-IDF)** with Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Tuned over n-gram range, `min_df`/`max_df`, `max_features`, `sublinear_tf`, smoothing (`alpha`), regularization (`C`), and class weighting.
2. **Static dense embeddings (FastText)** — word vectors combined into a document vector via **mean pooling**, then classified with NB / LogReg / LinearSVC.
3. **Learned sequence compression (BiLSTM)** in PyTorch — embedding layer seeded with FastText, an order-aware bidirectional LSTM replacing fixed pooling, and a weighted loss for the training imbalance.
4. **Contextual dense embeddings (SBERT, `all-mpnet-base-v2`)** — frozen sentence encoder producing 768-dim vectors, classified with Logistic Regression.
5. **Fine-tuned transformer encoder (RoBERTa-base)** — the full transformer trained end-to-end with a fresh 77-way classification head, on Apple Silicon (MPS).

### Imbalance handling

Because the training set is imbalanced but the test set is balanced, each model uses the appropriate lever: `fit_prior=False` / `sample_weight` for Naive Bayes, `class_weight="balanced"` for LogReg and SVM, and a frequency-weighted `CrossEntropyLoss` for the LSTM. Macro metrics are used throughout to expose any rare-class degradation.

## 📈 Results

| Approach | Representation | Classifier | Test Accuracy | Test Macro-F1 |
| :--- | :--- | :--- | :--- | :--- |
| TF-IDF | Sparse lexical | Multinomial NB | 0.869 | 0.867 |
| FastText (mean-pooled) | Static dense | LogReg / LinearSVC | *below TF-IDF* | *below TF-IDF* |
| BiLSTM | Learned sequence | — | 0.838 | 0.838 |
| **SBERT** (`all-mpnet-base-v2`) | **Contextual dense (frozen)** | **LogReg** | **0.937** | **0.937** |
| Fine-tuned RoBERTa | Contextual (fine-tuned) | end-to-end | 0.918 | 0.918 |

*(TF-IDF NB validated with 5-fold CV: macro-F1 0.852 ± 0.006; bootstrap 95% CI on test macro-F1 ≈ [0.855, 0.879].)*

### Key findings

* **Representation dominates the classifier.** Holding the classifier simple and upgrading only the representation drove every meaningful gain — the classifier was rarely the bottleneck.
* **Well-tuned TF-IDF is a strong baseline (~0.87).** On this fine-grained, keyword-driven task, sparse lexical matching beats *both* static-embedding averaging and a from-scratch LSTM.
* **Mean-pooling destroys signal.** Averaging FastText word vectors blurs the decisive keywords that distinguish similar intents, so it underperforms TF-IDF despite being "denser."
* **Contextual embeddings win.** Frozen SBERT + Logistic Regression reached **0.937**, the best result — because the encoder reads words in context and pools them in a learned way, recovering the distinctions that naive averaging lost.
* **The frozen specialist beat the fine-tuned generalist.** Fine-tuned RoBERTa (0.918) landed *below* frozen SBERT (0.937). SBERT was already trained on ~1B sentence pairs to produce excellent embeddings, whereas RoBERTa had to learn banking representations from scratch on ~9k examples in 3 epochs. The best representation here required no fine-tuning of our own. (RoBERTa was still improving at epoch 3 — more epochs/data would likely narrow or reverse this gap.)

## 📦 Stack

Polars · scikit-learn · gensim (FastText) · PyTorch · sentence-transformers · Hugging Face Transformers

## 📌 TODO — Zero-Shot Classification

Explore whether the **label names themselves** can drive classification via semantic similarity, using **no training labels at all**. The idea: embed each of the 77 intent names into the same SBERT space as the queries, then predict each query by its nearest label (cosine similarity). This tests whether the label text is a strong enough semantic description to match queries without any supervised training — a true zero-shot baseline. Expected to score below the trained models (label names are terse and several intents overlap), but valuable as a "no training data" reference point and for handling new intents without retraining. Label wording/engineering (e.g. expanding `card_arrival` → *"tracking when my new card will arrive"*) is likely to matter more here than anywhere else.