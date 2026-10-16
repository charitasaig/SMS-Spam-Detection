"""
Model Training & Comparison Pipeline for SMS Spam Message Detection.
Team 2 - Natural Language Processing

This script:
1. Loads the UCI SMS Spam dataset via src.data_loader.
2. Applies stratified train-test split (80/20) with fixed random seed (42).
3. Builds Scikit-learn Pipelines with TF-IDF Vectorizer (unigrams + bigrams, max_features=5000).
4. Trains:
   - Multinomial Naive Bayes (MultinomialNB)
   - Logistic Regression (LogisticRegression)
5. Evaluates both models on the unseen test set using real metrics.
6. Selects the superior model objectively based on Spam F1-Score & Spam Precision.
7. Serializes the trained pipelines and evaluation metrics to the models/ directory.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_sms_data
from src.preprocessing import clean_text
from src.evaluate_model import compute_model_metrics, format_comparison_table


def build_pipeline(classifier, ngram_range=(1, 2), max_features=5000) -> Pipeline:
    """
    Constructs an end-to-end Scikit-learn Pipeline with text preprocessing and TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])


def train_and_evaluate_models(random_state: int = 42, test_size: float = 0.2):
    """
    Orchestrates the entire training, evaluation, comparison, and saving workflow.
    """
    print("=" * 70)
    print(" SMS SPAM MESSAGE DETECTION - TRAINING & EVALUATION PIPELINE")
    print(" Team 2 | Natural Language Processing")
    print("=" * 70)

    # 1. Load dataset
    print("\n[Step 1/6] Loading UCI SMS Spam Collection dataset...")
    df, dataset_stats = load_sms_data(drop_duplicates=True, auto_download=True)
    print(f"-> Total clean messages loaded: {len(df):,}")
    print(f"-> Class distribution: Ham = {dataset_stats['final_distribution']['ham']:,} "
          f"({dataset_stats['final_distribution']['ham_percentage']}%), "
          f"Spam = {dataset_stats['final_distribution']['spam']:,} "
          f"({dataset_stats['final_distribution']['spam_percentage']}%)")

    # 2. Stratified train-test split
    print("\n[Step 2/6] Performing stratified train-test split (80% train, 20% test)...")
    X = np.array(df["message"].tolist(), dtype=object)
    y = np.array(df["label_num"].tolist(), dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"-> Training samples: {len(X_train):,} (Ham: {int((y_train == 0).sum())}, Spam: {int((y_train == 1).sum())})")
    print(f"-> Testing samples:  {len(X_test):,} (Ham: {int((y_test == 0).sum())}, Spam: {int((y_test == 1).sum())})")

    # 3. Model 1: Multinomial Naive Bayes Pipeline
    print("\n[Step 3/6] Training Model 1: Multinomial Naive Bayes Pipeline...")
    nb_pipeline = build_pipeline(MultinomialNB(alpha=0.2), ngram_range=(1, 2), max_features=5000)
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    nb_proba = nb_pipeline.predict_proba(X_test)
    nb_metrics = compute_model_metrics(y_test, nb_pred, nb_proba, model_name="Multinomial Naive Bayes")
    print(f"-> Naive Bayes Test Accuracy:      {nb_metrics['accuracy'] * 100:.2f}%")
    print(f"-> Naive Bayes Spam F1-Score:       {nb_metrics['spam_f1'] * 100:.2f}%")
    print(f"-> Naive Bayes Spam Precision:      {nb_metrics['spam_precision'] * 100:.2f}%")
    print(f"-> Naive Bayes Spam Recall:         {nb_metrics['spam_recall'] * 100:.2f}%")
    print(f"-> Naive Bayes Confusion Matrix:    TN={nb_metrics['confusion_matrix']['tn']}, "
          f"FP={nb_metrics['confusion_matrix']['fp']}, FN={nb_metrics['confusion_matrix']['fn']}, "
          f"TP={nb_metrics['confusion_matrix']['tp']}")

    # 4. Model 2: Logistic Regression Pipeline
    print("\n[Step 4/6] Training Model 2: Logistic Regression Pipeline...")
    lr_pipeline = build_pipeline(
        LogisticRegression(C=5.0, max_iter=1000, random_state=random_state, solver="liblinear"),
        ngram_range=(1, 2),
        max_features=5000
    )
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)
    lr_metrics = compute_model_metrics(y_test, lr_pred, lr_proba, model_name="Logistic Regression")
    print(f"-> Logistic Regression Test Accuracy:  {lr_metrics['accuracy'] * 100:.2f}%")
    print(f"-> Logistic Regression Spam F1-Score:   {lr_metrics['spam_f1'] * 100:.2f}%")
    print(f"-> Logistic Regression Spam Precision:  {lr_metrics['spam_precision'] * 100:.2f}%")
    print(f"-> Logistic Regression Spam Recall:     {lr_metrics['spam_recall'] * 100:.2f}%")
    print(f"-> Logistic Regression Confusion Matrix: TN={lr_metrics['confusion_matrix']['tn']}, "
          f"FP={lr_metrics['confusion_matrix']['fp']}, FN={lr_metrics['confusion_matrix']['fn']}, "
          f"TP={lr_metrics['confusion_matrix']['tp']}")

    # 5. Objective Model Selection
    print("\n[Step 5/6] Comparing models based on empirical test results...")
    comparison_table = format_comparison_table([nb_metrics, lr_metrics])
    print("\nComparison Summary:")
    print(pd.DataFrame(comparison_table).to_string(index=False))

    # Selection logic:
    # Primary metric: Spam F1-Score (harmonic mean of Precision and Recall on the minority spam class)
    # Tie-breaker: Spam Precision (prioritizing low False Positives to avoid marking legitimate SMS as spam)
    if nb_metrics["spam_f1"] > lr_metrics["spam_f1"]:
        best_name = "Multinomial Naive Bayes"
        best_pipeline = nb_pipeline
        best_metrics = nb_metrics
        reason = (f"Multinomial Naive Bayes achieved a higher Spam F1-Score "
                  f"({nb_metrics['spam_f1']*100:.2f}% vs {lr_metrics['spam_f1']*100:.2f}%).")
    elif lr_metrics["spam_f1"] > nb_metrics["spam_f1"]:
        best_name = "Logistic Regression"
        best_pipeline = lr_pipeline
        best_metrics = lr_metrics
        reason = (f"Logistic Regression achieved a higher Spam F1-Score "
                  f"({lr_metrics['spam_f1']*100:.2f}% vs {nb_metrics['spam_f1']*100:.2f}%).")
    else:
        # Tie break on precision (higher precision = fewer false positives)
        if nb_metrics["spam_precision"] >= lr_metrics["spam_precision"]:
            best_name = "Multinomial Naive Bayes"
            best_pipeline = nb_pipeline
            best_metrics = nb_metrics
            reason = "Tied F1-Score; Multinomial Naive Bayes selected for superior Spam Precision (fewer false positives)."
        else:
            best_name = "Logistic Regression"
            best_pipeline = lr_pipeline
            best_metrics = lr_metrics
            reason = "Tied F1-Score; Logistic Regression selected for superior Spam Precision (fewer false positives)."

    print(f"\n★ Selected Champion Model: {best_name}")
    print(f"  Rationale: {reason}")

    # 6. Save artifacts
    print("\n[Step 6/6] Saving trained pipelines, evaluation artifacts, and comparison data...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save pipelines
    best_model_path = os.path.join(models_dir, "best_model_pipeline.joblib")
    nb_model_path = os.path.join(models_dir, "naive_bayes_pipeline.joblib")
    lr_model_path = os.path.join(models_dir, "logistic_regression_pipeline.joblib")

    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(nb_pipeline, nb_model_path)
    joblib.dump(lr_pipeline, lr_model_path)
    print(f"-> Saved champion pipeline to: {best_model_path}")
    print(f"-> Saved Naive Bayes pipeline to: {nb_model_path}")
    print(f"-> Saved Logistic Regression pipeline to: {lr_model_path}")

    # Save metrics and metadata
    metrics_payload = {
        "best_model_name": best_name,
        "selection_rationale": reason,
        "models": {
            "naive_bayes": nb_metrics,
            "logistic_regression": lr_metrics
        },
        "comparison_table": comparison_table,
        "split_info": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_spam": int((y_train == 1).sum()),
            "train_ham": int((y_train == 0).sum()),
            "test_spam": int((y_test == 1).sum()),
            "test_ham": int((y_test == 0).sum())
        },
        "dataset_stats": dataset_stats
    }

    metrics_json_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"-> Saved comprehensive metrics JSON to: {metrics_json_path}")

    # Save test set predictions and raw test messages for dashboard drill-down
    eval_cache_path = os.path.join(models_dir, "evaluation_cache.joblib")
    joblib.dump({
        "X_test": X_test,
        "y_test": y_test,
        "nb_pred": nb_pred,
        "nb_proba": nb_proba,
        "lr_pred": lr_pred,
        "lr_proba": lr_proba
    }, eval_cache_path)
    print(f"-> Saved evaluation test cache to: {eval_cache_path}")

    print("\n" + "=" * 70)
    print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_models()
