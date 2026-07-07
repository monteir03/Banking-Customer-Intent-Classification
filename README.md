# BANKING77
## 📊 Dataset & Task Description

This project utilizes the **BANKING77** dataset (originally curated by PolyAI) to build and evaluate an intent classification system for financial customer support.

### 👥 Dataset Overview
BANKING77 is a specialized, fine-grained text classification benchmark comprising real-world customer service queries from the banking domain. 

* **Total Samples:** 13,083 customer utterances.
* **Train/Test Split:** 10,003 training samples and 3,080 evaluation samples.
* **Granularity:** 77 distinct, fine-grained intent classes.

### 🎯 The Machine Learning Task
The core task is **Single-Label, Multi-Class Text Classification**. Given a raw text input from a user (e.g., a chat message or support ticket), the model must predict exactly one correct intent out of the 77 possible categories.

#### Key Technical Challenges:
* **Fine-Grained Semantic Overlap:** Many categories share almost identical vocabulary and differ only by subtle context. For example, distinguishing between `card_linking`, `card_acceptance`, and `card_delivery_estimate` requires high semantic precision.
* **Real-World Noise:** The text inputs contain natural human variations, including casual slang, typos, shorthand, and varying sentence lengths.
* **High-Dimensional Class Space:** Managing a 77-way classification split introduces a high risk of misclassification compared to simpler binary or low-class tasks.

#### Example Mappings:

| Customer Query | Ground Truth Intent (Label) |
| :--- | :--- |
| *"I think my card was stolen, can you block it?"* | `card_loss_reported` |
| *"Why was I charged an extra fee at the cash machine?"* | `atm_support` |
| *"Can I link my account to my phone for wireless payments?"* | `apple_pay_or_google_pay` |

### 🛠️ Data Engineering Pipeline (Polars)
To ensure high-performance data loading and manipulation, local Parquet files are managed using the **Polars** library. This allows for lightning-fast out-of-memory schema validation before passing data into the Hugging Face Tokenization pipeline.

