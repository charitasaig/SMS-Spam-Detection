# SMS Spam Message Detection Using NLP & Machine Learning

**Subject:** Natural Language Processing (NLP)  
**Team Number:** Team 2  
**Academic Level:** College / Undergraduate Project  
**Core Technologies:** Python 3.12, Scikit-learn, Streamlit, Pandas, NumPy, Joblib  

---

## 📌 Project Overview
This project delivers a complete, production-grade Natural Language Processing (NLP) and Machine Learning solution to classify incoming SMS text messages into:
1. **HAM** — Legitimate personal, social, or transactional communication.
2. **SPAM** — Unwanted, promotional, or potentially suspicious communication.

Built using the authentic **UCI SMS Spam Collection dataset**, this system implements specialized domain-aware text preprocessing (preserving URLs, currency tokens, phone numbers, and grammatical negations), TF-IDF n-gram feature extraction, empirical model comparison between **Multinomial Naive Bayes** and **Logistic Regression**, and an interactive, responsive **Streamlit Web Application**.

---

## 🏆 Measured Model Benchmark Results
Both models were evaluated on the identical stratified 20% test split ($N = 1,032$ unseen messages):

| Metric | Multinomial Naive Bayes | Logistic Regression (Champion) | Winner / Notes |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 98.35% | **98.64%** | **Logistic Regression** (+0.29%) |
| **Spam Precision** | 97.44% | **100.00%** | **Logistic Regression (Zero False Positives!)** |
| **Spam Recall** | 89.06% | 89.06% | Tie |
| **Spam F1-Score** | 93.06% | **94.21%** | **Logistic Regression** (+1.15%) |
| **Ham F1-Score** | 99.07% | **99.23%** | **Logistic Regression** |
| **False Positives (FP)** | 3 | **0** | **Logistic Regression (0 legitimate msgs blocked)** |
| **False Negatives (FN)** | 14 | 14 | Both missed 14 out of 128 test spam |

> **Champion Selection Rationale**: Logistic Regression achieved a higher Spam F1-Score (94.21%) and **100.00% Spam Precision with 0 False Positives**, ensuring no critical legitimate communications (such as bank OTPs or family messages) are mistakenly marked as spam.

---

## 📂 Project Architecture
```
sms-spam-detection/
│
├── data/
│   ├── README.md                  # Dataset provenance, download guide, and metadata
│   └── SMSSpamCollection          # Authentic UCI dataset (auto-downloaded or local)
│
├── models/
│   ├── .gitkeep                   # Version control placeholder
│   ├── best_model_pipeline.joblib # Serialized champion pipeline (Logistic Regression)
│   ├── naive_bayes_pipeline.joblib# Serialized Multinomial Naive Bayes pipeline
│   ├── logistic_regression_pipeline.joblib # Serialized Logistic Regression pipeline
│   ├── model_metrics.json         # Real empirical evaluation metrics & split stats
│   └── evaluation_cache.joblib    # Cached test-set predictions for instant drill-down
│
├── notebooks/
│   └── sms_spam_analysis.ipynb    # Complete step-by-step Jupyter analysis & EDA
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # Automatic downloader, cleaner, and deduplicator
│   ├── preprocessing.py          # Domain NLP preprocessor with negation preservation
│   ├── train_model.py             # Stratified pipeline training, evaluation & saving
│   ├── evaluate_model.py          # Metric calculations, confusion matrices & reports
│   ├── predict.py                 # Safe inference engine with probability calibration
│   └── test_scenarios.py          # Verification suite testing 8 mandatory scenarios
│
├── app.py                         # 5-Page Streamlit Web Application
├── requirements.txt               # Pinned, tested Python dependencies
├── README.md                      # Complete setup and execution guide (Windows)
├── .gitignore                     # Git ignore rules for virtual environments & cache
└── project_report.md              # 19-section academic college project report
```

---

## 🚀 Step-by-Step Setup & Execution Guide (Windows & VS Code)

Follow these exact steps in **VS Code** on Windows (PowerShell or Command Prompt):

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File -> Open Folder...** and select the folder:
   ```text
   D:\SMS Spam Message Detection
   ```
3. Open a new terminal in VS Code: **Terminal -> New Terminal** (or press `` Ctrl + ` ``).

---

### Step 2: Create and Activate a Virtual Environment
In your VS Code terminal (PowerShell), run:

```powershell
# Create virtual environment named 'venv'
python -m venv venv

# If PowerShell restricts script execution, run this once:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

*(If using Command Prompt `cmd.exe` instead, activate via: `venv\Scripts\activate.bat`)*

You will see `(venv)` appear at the start of your terminal prompt.

---

### Step 3: Install Required Dependencies
Upgrade pip and install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### Step 4: Train and Evaluate the Models
Run the training script:

```powershell
python src/train_model.py
```

**What this does automatically:**
1. Checks for the dataset; if missing, automatically downloads the official UCI zip archive.
2. Cleans the data, drops 414 duplicate messages, and validates the 5,160 clean unique messages.
3. Performs stratified 80/20 train-test splitting (`random_state=42`).
4. Fits both **Multinomial Naive Bayes** and **Logistic Regression** with TF-IDF unigrams and bigrams.
5. Computes actual test-set metrics, selects the champion model, and serializes all pipelines into `models/`.

---

### Step 5: Run the Verification Test Suite
Verify that all 8 mandatory test cases pass and inspect their genuine outputs:

```powershell
python src/test_scenarios.py
```

---

### Step 6: Launch the Streamlit Web Application
Start the interactive dashboard:

```powershell
python -m streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`.

---

## 🌐 Streamlit Web Application Features
The web application includes 5 comprehensive pages:
1. **🏠 Home Dashboard**: Problem overview, dataset metrics, class balance charts, and champion model KPIs.
2. **🔍 SMS Spam Detector**: Real-time message analyzer with sample loaders (phishing, lottery, bank, personal), probability breakdown, detected linguistic triggers, and non-definitive disclaimer.
3. **📊 Model Comparison**: Side-by-side empirical metrics table, interactive confusion matrices for both models, and deep-dive explanations of False Positives vs. False Negatives.
4. **📁 Dataset Explorer**: Interactive searchable viewer of the authentic UCI dataset with character-length and word-count distribution charts.
5. **ℹ️ About Project**: Architecture diagram, Team 2 credentials, ethical considerations, and future scope.

---

## 📓 Running the Jupyter Notebook
To run the step-by-step exploratory notebook in VS Code:
1. Open `notebooks/sms_spam_analysis.ipynb` in VS Code.
2. Select your `venv` Python environment as the Jupyter kernel in the top-right corner.
3. Click **Run All** to execute each step with inline plots and confusion matrices.

---

## ⚠️ Ethical & Academic Disclaimer
Predictions generated by this application are algorithmic estimations based on statistical lexical patterns. A **SPAM** classification indicates linguistic similarity to unsolicited or suspicious messaging, **not definitive legal proof of fraudulent or criminal intent**. Automated tools should never be used as sole proof in legal or security determinations.

---

## 👥 Authors
- **Team Number:** Team 2
- **Subject:** Natural Language Processing (NLP)
- **Institution:** College of Computer Science & Engineering
