"""
Data Loader Module for SMS Spam Message Detection.
Team 2 - Natural Language Processing

Responsible for:
- Downloading the official UCI SMS Spam Collection dataset.
- Loading local copies from data/SMSSpamCollection or data/spam.csv.
- Cleaning, identifying columns, handling missing values and duplicate messages.
- Returning clean DataFrames and metadata for training and exploration.
"""

import os
import zipfile
import io
import urllib.request
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Constants
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
GITHUB_BACKUP_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_RAW_FILE = os.path.join(DEFAULT_DATA_DIR, "SMSSpamCollection")


def download_uci_dataset(dest_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Downloads the official UCI SMS Spam Collection zip archive and extracts SMSSpamCollection.
    If the UCI static server is unavailable, falls back to the identical raw mirror.
    
    Args:
        dest_dir: Directory where the dataset file should be placed.
        
    Returns:
        Path to the extracted SMSSpamCollection file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    target_path = os.path.join(dest_dir, "SMSSpamCollection")
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
        print(f"[DataLoader] Dataset already exists locally at: {target_path}")
        return target_path

    print(f"[DataLoader] Attempting to download official UCI dataset from: {UCI_ZIP_URL}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        req = urllib.request.Request(UCI_ZIP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as response:
            zip_content = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                # Find SMSSpamCollection in the zip
                for filename in z.namelist():
                    if "SMSSpamCollection" in filename:
                        with z.open(filename) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        print(f"[DataLoader] Successfully extracted {target_path}")
                        return target_path
    except Exception as e:
        print(f"[DataLoader] Primary UCI zip download failed ({e}). Trying verified backup mirror...")
        
    try:
        req = urllib.request.Request(GITHUB_BACKUP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as response:
            content = response.read()
            with open(target_path, "wb") as f:
                f.write(content)
            print(f"[DataLoader] Successfully downloaded verified mirror to: {target_path}")
            return target_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to automatically download dataset from both UCI and backup mirrors: {e}.\n"
            f"Please download 'SMSSpamCollection' manually and place it at: {target_path}\n"
            f"Refer to {os.path.join(dest_dir, 'README.md')} for instructions."
        )


def load_sms_data(
    file_path: Optional[str] = None,
    drop_duplicates: bool = True,
    auto_download: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads, cleans, and validates the SMS Spam Collection dataset.
    
    Args:
        file_path: Optional specific path to SMSSpamCollection or csv file.
        drop_duplicates: Whether to drop identical repeated messages.
        auto_download: Whether to attempt automatic download if file is not found.
        
    Returns:
        (df, stats_dict)
        - df: pd.DataFrame with standardized columns ['label', 'message', 'label_num']
        - stats_dict: Dictionary containing summary statistics and counts.
    """
    if file_path is None:
        file_path = DEFAULT_RAW_FILE
        
    if not os.path.exists(file_path):
        # Check alternative common names in data/ directory
        data_dir = os.path.dirname(file_path)
        candidates = [
            os.path.join(data_dir, "SMSSpamCollection"),
            os.path.join(data_dir, "SMSSpamCollection.txt"),
            os.path.join(data_dir, "spam.csv"),
            os.path.join(data_dir, "sms.tsv")
        ]
        found = None
        for cand in candidates:
            if os.path.exists(cand) and os.path.getsize(cand) > 1000:
                found = cand
                break
                
        if found:
            file_path = found
        elif auto_download:
            file_path = download_uci_dataset(data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. Set auto_download=True or place the file manually."
            )

    # Read tab-separated or comma-separated file
    try:
        # Standard UCI format: tab-separated, no header, col 0 = label, col 1 = text
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1",
            quoting=3,  # csv.QUOTE_NONE prevents issues with unbalanced quotes in SMS text
            on_bad_lines="skip"
        )
    except Exception:
        # Fallback for CSV formats (e.g., Kaggle repackaging of UCI SMS)
        df = pd.read_csv(file_path, encoding="latin-1")
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
        elif "Category" in df.columns and "Message" in df.columns:
            df = df[["Category", "Message"]].rename(columns={"Category": "label", "Message": "message"})

    # Ensure required columns exist
    if "label" not in df.columns or "message" not in df.columns:
        # Use first two columns
        df = df.iloc[:, [0, 1]]
        df.columns = ["label", "message"]

    raw_count = len(df)
    
    # 1. Clean labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    
    # 2. Handle missing/null messages
    df["message"] = df["message"].astype(str).str.strip()
    missing_count = int(df["message"].isna().sum() + (df["message"] == "").sum())
    df = df[df["message"] != ""].copy()
    
    # 3. Duplicate handling
    duplicates_count = int(df.duplicated(subset=["message"]).sum())
    raw_ham_count = int((df["label"] == "ham").sum())
    raw_spam_count = int((df["label"] == "spam").sum())

    if drop_duplicates:
        df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
        
    cleaned_count = len(df)
    final_ham_count = int((df["label"] == "ham").sum())
    final_spam_count = int((df["label"] == "spam").sum())
    
    # Add numerical target label (0 = ham, 1 = spam)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    
    stats: Dict[str, Any] = {
        "source_file": file_path,
        "raw_messages_count": raw_count,
        "missing_messages_dropped": missing_count,
        "duplicates_count": duplicates_count,
        "final_messages_count": cleaned_count,
        "dropped_duplicates": drop_duplicates,
        "raw_distribution": {
            "ham": raw_ham_count,
            "spam": raw_spam_count,
            "spam_percentage": round((raw_spam_count / raw_count) * 100, 2) if raw_count else 0
        },
        "final_distribution": {
            "ham": final_ham_count,
            "spam": final_spam_count,
            "spam_percentage": round((final_spam_count / cleaned_count) * 100, 2) if cleaned_count else 0,
            "ham_percentage": round((final_ham_count / cleaned_count) * 100, 2) if cleaned_count else 0
        }
    }
    
    return df, stats


if __name__ == "__main__":
    print("[Testing data_loader.py directly]")
    df_data, info = load_sms_data(auto_download=True)
    print(f"Loaded {len(df_data)} messages.")
    print(f"Dataset stats: {info['final_distribution']}")
    print("Sample messages:")
    print(df_data.head(3))
