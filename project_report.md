# SMS Spam Message Detection Using Natural Language Processing (NLP) & Machine Learning

**Comprehensive Project Report**  
**Team Number:** Team 2  
**Subject:** Natural Language Processing (NLP)  
**Academic Level:** College / Undergraduate Project  
**Date of Submission:** October 2026  

---

## Table of Contents
1. [Title Page & Team Details](#1-title-page--team-details)
2. [Abstract](#2-abstract)
3. [Introduction to Natural Language Processing](#3-introduction-to-natural-language-processing)
4. [Problem Statement](#4-problem-statement)
5. [Objectives](#5-objectives)
6. [Dataset Description & Provenance](#6-dataset-description--provenance)
7. [NLP Preprocessing Techniques](#7-nlp-preprocessing-techniques)
8. [Feature Extraction (TF-IDF)](#8-feature-extraction-tf-idf)
9. [Theoretical Foundation of Machine Learning Models](#9-theoretical-foundation-of-machine-learning-models)
10. [System Architecture & Workflow](#10-system-architecture--workflow)
11. [Implementation Details](#11-implementation-details)
12. [Experimental Results & Comparative Analysis](#12-experimental-results--comparative-analysis)
13. [Confusion Matrix & Classification Reports](#13-confusion-matrix--classification-reports)
14. [Sample Predictions & Real-Time Demonstrations](#14-sample-predictions--real-time-demonstrations)
15. [Testing, Validation & Error Analysis](#15-testing-validation--error-analysis)
16. [Project Limitations](#16-project-limitations)
17. [Future Enhancements](#17-future-enhancements)
18. [Conclusion](#18-conclusion)
19. [References](#19-references)

---

## 1. Title Page & Team Details

| Project Attribute | Details |
| :--- | :--- |
| **Project Title** | SMS Spam Message Detection Using NLP and Machine Learning |
| **Team Number** | Team 2 |
| **Subject** | Natural Language Processing |
| **Programming Language** | Python 3.12 |
| **Core Libraries** | Scikit-learn, Pandas, NumPy, Streamlit, Matplotlib, Seaborn, Joblib |
| **Development Tools** | Visual Studio Code, Jupyter Notebook |
| **Dataset** | UCI SMS Spam Collection Repository |

---

## 2. Abstract
Short Message Service (SMS) remains one of the world's most pervasive and trusted point-to-point communication channels, boasting open rates exceeding 90%. However, this elevated level of user trust makes mobile messaging a prime target for adversarial threat actors deploying unsolicited advertising, smishing (SMS phishing), lottery scams, and financial identity theft.

This project presents an end-to-end, reproducible NLP and machine learning pipeline developed by **Team 2** to classify raw SMS messages into legitimate communication (**HAM**) and unsolicited or suspicious messages (**SPAM**). Utilizing the authentic UCI SMS Spam Collection dataset (5,574 messages), we address class imbalance and data hygiene by eliminating 414 duplicate entries and preserving critical spam indicators (such as currency symbols, numeric shortcodes, and URLs) along with grammatical negations. We engineer feature representations using TF-IDF (Term Frequency-Inverse Document Frequency) unigram and bigram tokenization and empirically benchmark **Multinomial Naive Bayes** against **Logistic Regression** across a stratified 80/20 train-test split.

On an unseen test set of 1,032 messages, **Logistic Regression** emerged as the champion classifier, attaining **98.64% overall accuracy**, a **94.21% Spam F1-Score**, and **100.00% Spam Precision (Zero False Positives)**, outperforming Multinomial Naive Bayes (98.35% accuracy, 93.06% Spam F1-Score, 3 False Positives). The final serialized pipeline is deployed via a responsive, interactive Streamlit web application providing live probability estimations, linguistic trigger explanations, comparative diagnostics, and dataset exploration.

---

## 3. Introduction to Natural Language Processing
Natural Language Processing (NLP) is an interdisciplinary domain combining computational linguistics, computer science, and machine learning to enable computing systems to analyze, understand, interpret, and generate human language.

Unlike tabular or numerical data, text is fundamentally unstructured, discrete, and characterized by high semantic variability, slang, typographical errors, and contextual ambiguity. In the context of SMS messaging:
- Text is constrained by length (typically 160 characters per SMS segment).
- Messages exhibit informal syntax, character omissions, phonological abbreviations (e.g., *u*, *gr8*, *txt*), and deliberate adversarial obfuscations.
- Automated classification requires systematic transformation of raw character strings into mathematically rigorous numerical vector spaces that preserve semantic meaning.

---

## 4. Problem Statement
Mobile subscribers are subjected to billions of unsolicited text messages monthly. While traditional email spam filters leverage extensive IP reputation networks, DKIM signatures, and multi-kilobyte header metadata, SMS messaging protocols provide minimal carrier metadata to recipient handsets.

The core challenge is to construct a lightweight, high-precision text classification algorithm capable of distinguishing between:
1. **HAM**: Legitimate personal, social, or transactional communications.
2. **SPAM**: Unwanted, unsolicited, or suspicious commercial and fraudulent transmissions.

### The Asymmetric Cost of Classification Errors:
- **False Positive (Type I Error)**: Classifying a legitimate message as spam. In SMS, this is critical because users may fail to receive time-sensitive Two-Factor Authentication (2FA) one-time passwords (OTPs), medical appointment alerts, flight updates, or emergency personal messages.
- **False Negative (Type II Error)**: Classifying a spam message as legitimate. The message appears in the user's inbox, causing minor inconvenience or requiring user vigilance to avoid clicking phishing links.

Consequently, model selection must penalize False Positives heavily, prioritizing high **Spam Precision**.

---

## 5. Objectives
1. **Data Acquisition & Hygiene**: Secure the official UCI SMS Spam Collection dataset, inspect message formatting, remove redundant duplicates, and analyze class distributions without fabricating any metrics.
2. **Specialized NLP Preprocessing**: Develop a domain-specific text preprocessing routine that standardizes input text while conscientiously preserving spam-indicative morphological patterns (currency signs, URLs, numbers, exclamation marks) and syntactic negations.
3. **Feature Extraction**: Construct a Scikit-learn TF-IDF pipeline supporting unigrams and bigrams with sublinear term-frequency scaling.
4. **Model Training & Comparison**: Train Multinomial Naive Bayes and Logistic Regression on identical stratified splits and select the superior model strictly based on empirical test metrics.
5. **Interactive Web Interface**: Deploy a Streamlit web application featuring a home dashboard, real-time message analyzer, model comparison diagnostics, and dataset explorer.
6. **Error Analysis & Ethics**: Thoroughly examine model misclassifications and emphasize non-definitive classification disclaimers.

---

## 6. Dataset Description & Provenance
- **Dataset Title**: UCI SMS Spam Collection
- **Source**: UCI Machine Learning Repository (`https://archive.ics.uci.edu/dataset/228/sms+spam+collection`)
- **Curators**: Tiago A. Almeida and José María Gómez Hidalgo (DocEng'11)
- **Original Format**: Tab-separated text (`SMSSpamCollection`), UTF-8/Latin-1 encoding.

### Dataset Cleaning & Deduplication Summary:
| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Raw Message Records** | 5,574 | 100.00% |
| **Missing / Null Messages** | 0 | 0.00% |
| **Duplicate Messages Identified & Removed** | 414 | 7.43% |
| **Clean Unique Corpus** | **5,160** | **92.57%** |
| **Final Legitimate (Ham) Messages** | 4,518 | 87.56% |
| **Final Spam Messages** | 642 | 12.44% |

**Note on Duplicates**: Common short messages like *"Ok"*, *"Sorry, I'll call later"*, or repeated marketing broadcasts appear multiple times in the raw collection. Retaining duplicates causes data leakage across train and test splits, artificially inflating evaluation metrics. Removing identical messages ensures genuine out-of-sample generalization.

---

## 7. NLP Preprocessing Techniques
Traditional text normalization pipelines frequently apply aggressive stop-word removal and punctuation stripping. In SMS spam classification, however, standard stripping destroys the very features that characterize unsolicited advertising.

Team 2 implemented a domain-aware preprocessing strategy in `src/preprocessing.py`:
1. **URL Normalization**: Regular expressions detect URLs (`http://`, `https://`, `www.`, `.com`, `.net`, etc.) and replace them with a unified token: `url_token`.
2. **Currency Symbol Tagging**: Symbols such as `£`, `$`, `€`, `₹` and terms like `cash` and `prize` are mapped to `currency_token`.
3. **Phone Number & Shortcode Tagging**: Sequences of digits exceeding 4 characters, international prefixes, and toll-free codes (e.g., `0800`, `0906`) are transformed to `number_token`.
4. **Exclamation Clustering**: Multiple exclamation points (e.g., `!!!`) are tagged as `exclamation_token`.
5. **Negation Preservation**: Words such as `not`, `no`, `never`, `none`, `neither`, `nor`, and `without` are explicitly preserved to protect grammatical intent.
6. **Selective Stop-Word Filtering**: Common grammatical particles (*the*, *is*, *at*, *which*) are pruned while keeping semantically critical modifiers.

---

## 8. Feature Extraction (TF-IDF)
To feed textual data into mathematical models, words are converted to numerical weights via **Term Frequency-Inverse Document Frequency (TF-IDF)**:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- $\text{TF}(t, d)$ represents the frequency of term $t$ in document $d$. We enable `sublinear_tf=True`, replacing raw count $f$ with $1 + \log(f)$ to dampen the influence of overly repetitive words.
- $\text{IDF}(t, D)$ is the inverse document frequency:
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **N-gram Range**: Configured as `ngram_range=(1, 2)` (unigrams and bigrams). Capturing word pairs (e.g., *"cash prize"*, *"call now"*, *"claim ur"*) provides strong contextual signals.
- **Max Features**: Capped at 5,000 top features to prevent overfitting and ensure fast inference.
- **Pipeline Encapsulation**: Fitted exclusively on training data within a Scikit-learn `Pipeline` to prevent data leakage.

---

## 9. Theoretical Foundation of Machine Learning Models

### 1. Multinomial Naive Bayes (MNB)
Multinomial Naive Bayes is a probabilistic classifier founded on Bayes' Theorem under the naive assumption of conditional feature independence:

$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)}{P(\mathbf{x})}$$

For text classification:
- $C_k \in \{\text{HAM}, \text{SPAM}\}$
- $P(x_i \mid C_k)$ is estimated with Laplace smoothing:
  $$P(x_i \mid C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha |V|}$$
  Where $\alpha = 0.2$ was selected for smoothing, $N_{ki}$ is the count of feature $i$ in class $k$, and $|V|$ is vocabulary size.
- **Strengths**: Extremely fast training, effective with sparse high-dimensional data, performs well even with modest dataset sizes.
- **Weaknesses**: The feature independence assumption ignores token correlations.

### 2. Logistic Regression (LR)
Logistic Regression is a linear discriminative classification model that estimates posterior class probability via the logistic sigmoid function:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

- **Optimization**: Minimizes binary cross-entropy loss with L2 regularization:
  $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{j=1}^{N} \left[ y_j \log(\hat{y}_j) + (1 - y_j) \log(1 - \hat{y}_j) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
  Configured with inverse regularization parameter $C = 5.0$, `solver='liblinear'`, and `random_state=42`.
- **Strengths**: Well-calibrated class probabilities, resilient to feature correlations, supports linear decision boundaries with zero bias toward balanced priors.
- **Weaknesses**: May require regularization tuning to prevent overfitting on small vocabularies.

---

## 10. System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                   UCI SMS SPAM DATASET                      │
│                  (5,574 Raw SMS Messages)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA LOADER (data_loader.py)                │
│    • Verify integrity, handle nulls, drop 414 duplicates    │
│    • Result: 5,160 Clean Unique Messages (87.6% Ham)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         STRATIFIED TRAIN-TEST SPLIT (80% / 20%)             │
│    • Training Set: 4,128 messages (Ham: 3614, Spam: 514)    │
│    • Test Set:     1,032 messages (Ham: 904,  Spam: 128)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            TF-IDF VECTORIZATION (preprocessing.py)          │
│    • Clean text, tag URLs/currency/numbers, keep negations  │
│    • Unigrams + Bigrams, max_features=5000                  │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│   MULTINOMIAL NAIVE BAYES    │ │     LOGISTIC REGRESSION      │
│      Pipeline (alpha=0.2)    │ │      Pipeline (C=5.0)        │
└──────────────┬───────────────┘ └─────────────┬────────────────┘
               │                               │
               ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│             EMPIRICAL EVALUATION & COMPARISON               │
│    • Accuracy, Precision, Recall, F1-Score, Confusion Matrix│
│    • Champion Selection: Logistic Regression (F1: 94.21%)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           SERIALIZATION & PERSISTENCE (Joblib/JSON)         │
│    • models/best_model_pipeline.joblib                      │
│    • models/model_metrics.json                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 STREAMLIT WEB APPLICATION                   │
│   Dashboard • Real-Time Detector • Benchmarks • Explorer    │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Implementation Details
The codebase is structured modularly inside `sms-spam-detection/`:
- `data/`: Contains dataset files and provenance documentation.
- `models/`: Stores serialized pipelines (`.joblib`) and metrics records (`.json`).
- `notebooks/sms_spam_analysis.ipynb`: Interactive Jupyter analysis with visualizations.
- `src/preprocessing.py`: Domain-specific text cleaning, regex tagging, and negation handling.
- `src/data_loader.py`: Dataset acquisition from UCI, deduplication, and schema validation.
- `src/evaluate_model.py`: Rigorous calculation of confusion matrices, precision, recall, and F1.
- `src/train_model.py`: End-to-end training, benchmarking, and champion model selection.
- `src/predict.py`: Safe inference API with probability calibration and trigger explanations.
- `app.py`: 5-page responsive Streamlit application.

---

## 12. Experimental Results & Comparative Analysis

Both models were fitted on the identical training set ($N=4,128$) and evaluated on the identical unseen test set ($N=1,032$).

### Measured Test-Set Performance Summary:
| Metric | Multinomial Naive Bayes | Logistic Regression (Champion) | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 98.35% | **98.64%** | +0.29% |
| **Spam Precision** | 97.44% | **100.00%** | **+2.56%** |
| **Spam Recall** | 89.06% | 89.06% | 0.00% |
| **Spam F1-Score** | 93.06% | **94.21%** | **+1.15%** |
| **Ham Precision** | 98.47% | 98.47% | 0.00% |
| **Ham Recall** | 99.67% | **100.00%** | +0.33% |
| **Ham F1-Score** | 99.07% | **99.23%** | +0.16% |
| **False Positives (FP)** | 3 | **0** | **-3 (Zero FP)** |
| **False Negatives (FN)** | 14 | 14 | 0 |

### Model Selection Decision:
**Logistic Regression** was selected as the champion model. While both models achieved identical recall on the spam class (89.06%), Logistic Regression achieved **100.00% Spam Precision with zero false positives** on the test set. In telephony environments, preventing the erroneous quarantine of legitimate messages is paramount.

---

## 13. Confusion Matrix & Classification Reports

### 1. Confusion Matrix Breakdown ($N=1,032$ Test Messages)

#### Multinomial Naive Bayes:
$$\begin{pmatrix} \text{TN} = 901 & \text{FP} = 3 \\ \text{FN} = 14 & \text{TP} = 114 \end{pmatrix}$$

#### Logistic Regression (Champion):
$$\begin{pmatrix} \text{TN} = 904 & \text{FP} = 0 \\ \text{FN} = 14 & \text{TP} = 114 \end{pmatrix}$$

### 2. Full Scikit-learn Classification Report (Logistic Regression)
```
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       904
        spam       1.00      0.89      0.94       128

    accuracy                           0.99      1032
   macro avg       0.99      0.95      0.97      1032
weighted avg       0.99      0.99      0.99      1032
```

---

## 14. Sample Predictions & Real-Time Demonstrations

The trained pipeline was evaluated against diverse representative SMS messages:

```
Sample 1: "Congratulations! You have been selected as our £1,000 weekly cash prize winner! Text CLAIM to 87012 or call 09061743811 to redeem."
-> Prediction: SPAM
-> Confidence: 99.92% (Spam Prob: 99.92%, Ham Prob: 0.08%)
-> Triggers: currency_token, number_token, exclamation_token

Sample 2: "Hey Mom, I am on my way home from college. Do we need anything from the grocery store?"
-> Prediction: HAM
-> Confidence: 99.54% (Spam Prob: 0.46%, Ham Prob: 99.54%)
-> Triggers: Conversational syntax, absence of promotional markers

Sample 3: "Meeting room #402, 10:30 AM tomorrow -- please bring 3 printed copies of doc #712! Thanks :)"
-> Prediction: HAM
-> Confidence: 97.80% (Spam Prob: 2.20%, Ham Prob: 97.80%)
-> Triggers: Internal work coordination despite punctuation and numeric tokens
```

---

## 15. Testing, Validation & Error Analysis

As mandated by Section 10, the complete application was tested against eight mandatory scenarios:

| Test ID | Category | Input Text | Prediction | Confidence | Key Observation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **TC-01** | Normal Personal | *"Hey Mom, I am on my way home from college..."* | **HAM** | 99.54% | Correctly identified as legitimate personal communication. |
| **TC-02** | Promotional SMS | *"EXCLUSIVE SALE! Get 50% discount on all designer jackets..."* | **HAM** | 59.68% | Conservative boundary in LR (Spam prob: 40.32%). MNB flagged as SPAM (71.53%). |
| **TC-03** | Suspicious Link | *"Urgent alert: Your postal package delivery could not be completed..."* | **HAM** | 60.17% | Modern phishing phrasing scored 39.83% spam prob in LR, but 83.24% SPAM in MNB. |
| **TC-04** | Prize / Reward | *"Congratulations! You have been selected as our £1,000 weekly cash prize..."* | **SPAM** | 99.92% | Overwhelming spam indicators correctly flagged. |
| **TC-05** | Bank Alert | *"Your checking account ending in 8419 was debited for $24.50... reply HELP."* | **SPAM** | 66.87% | **False Positive Error**: Triggered by multiple `$` symbols and numeric account numbers. |
| **TC-06** | Empty Input | *"   "* | **INVALID** | 0.00% | Validation caught whitespace input cleanly without crashing. |
| **TC-07** | Numbers & Punctuation | *"Meeting room #402, 10:30 AM tomorrow... doc #712! Thanks :)"* | **HAM** | 97.80% | Punctuation and room digits did not trigger a false alarm. |
| **TC-08** | Mixed Context | *"Hey John, did you hear about that free pizza offer...?"* | **HAM** | 93.95% | Casual syntax successfully overrode single keyword *"free"*. |

### Detailed Error Analysis on TC-05 (The Bank Notification False Positive):
TC-05 was predicted as SPAM (66.87% confidence). Why?
In the 2011 UCI dataset, dollar signs (`$`) and terms like *"cash"*, *"debited"*, and *"reply HELP"* appeared almost exclusively in commercial sweepstakes spam. Legitimate bank SMS alerts adhering to modern US financial notification standards were non-existent in the UK/Singapore 2007-2011 dataset. 

This error provides two crucial academic takeaways:
1. **Concept Drift & Corpus Age**: Machine learning models reflect the statistical distribution of their training data. A model trained on 2011 texts may over-index on currency symbols.
2. **The Imperative for Disclaimers**: This false positive proves why automated tools must **never** claim an SMS is definitively fraudulent. Real-world systems require human review or whitelists for registered bank shortcodes.

---

## 16. Project Limitations
1. **Corpus Timeframe**: The UCI dataset was compiled in 2011. While core spam patterns (urgency, prizes, links) remain consistent, modern smishing incorporates package delivery scams, cryptocurrency invitations, and MFA fatigue attacks.
2. **Language Scope**: The collection comprises English SMS. Messages in Spanish, Hindi, or mixed codes (e.g., Hinglish) require multilingual embedding models.
3. **Absence of Telephony Metadata**: In production telecommunications networks, classifiers inspect carrier routing, sender registration (e.g., Alpha Sender IDs), and message velocity. Our NLP model operates strictly on textual content.

---

## 17. Future Enhancements
1. **Pretrained Transformer Embeddings**: Fine-tuning lightweight transformer models like **DistilBERT** or **DeBERTa** to capture deeper bidirectional contextual dependencies.
2. **Sender Whitelisting**: Integrating verified bank and delivery sender shortcode registries to eliminate false positives on transactional SMS.
3. **Active Learning & Feedback Loop**: Allowing users to flag misclassified messages to retrain the pipeline on edge cases continuously.
4. **On-Device Inference**: Quantizing the model (e.g., ONNX / TFLite) for on-device SMS spam filtering without compromising mobile privacy.

---

## 18. Conclusion
The **SMS Spam Message Detection Project** developed by **Team 2** demonstrates a complete, academically rigorous, and practical application of Natural Language Processing and Machine Learning. By conscientiously addressing data hygiene, engineering indicator-preserving text preprocessing, and benchmarking across stratified splits, we produced a champion **Logistic Regression pipeline** achieving **98.64% accuracy** and **zero false positives** on the test dataset.

The deployed Streamlit web application provides an intuitive demonstration platform, bridging machine learning theory with transparent, explainable software design.

---

## 19. References
1. Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng'11)*, Mountain View, CA, USA.
2. UCI Machine Learning Repository (2012). *SMS Spam Collection Dataset*. Available at: https://archive.ics.uci.edu/dataset/228/sms+spam+collection.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825-2830.
4. Jurafsky, D., & Martin, J.H. (2024). *Speech and Language Processing (3rd ed. draft)*. Stanford University.
5. Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
