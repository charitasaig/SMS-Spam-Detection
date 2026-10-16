# UCI SMS Spam Collection Dataset

This folder contains the dataset used for training, evaluating, and demonstrating the SMS Spam Message Detection system.

## Dataset Source
- **Official Source**: [UCI Machine Learning Repository: SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Creators / Curators**: Tiago A. Almeida and José María Gómez Hidalgo
- **Original Publication**: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng'11)*, Mountain View, CA, USA, 2011.

## Format & Attributes
The dataset is composed of 5,574 English SMS messages tagged according to whether they are legitimate (ham) or spam:
- **Number of Instances**: 5,574 messages
- **Columns**:
  1. `label`: Class label (`ham` or `spam`)
  2. `message`: Raw text of the SMS message
- **Format**: Tab-separated values (`\t`), UTF-8 encoding. Each line begins with the label followed by a tab and the SMS text.

## Class Distribution (Raw Dataset)
- **Ham (legitimate)**: 4,825 messages (~86.6%)
- **Spam (unwanted/suspicious)**: 747 messages (~13.4%)

## Automated Acquisition
The data loader (`src/data_loader.py`) will automatically download and extract the dataset from UCI's repository (`https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`) if the file is not found locally.

## Manual Placement
If your environment does not have internet access or the UCI server is unreachable:
1. Download `sms+spam+collection.zip` from https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Extract the file `SMSSpamCollection` into this `data/` folder.
3. The expected file path is `data/SMSSpamCollection`.

## Dataset Limitations & Ethical Considerations
1. **Language & Regional Bias**: Messages are primarily in English, collected in the late 2000s and early 2010s in the UK and Singapore. Modern spam may contain different slang, regional dialects, or multi-language elements.
2. **Short Length**: SMS messages are constrained (often <= 160 characters), meaning contextual information can be sparse.
3. **Imbalance**: Ham messages heavily outnumber spam messages (~6.5 : 1), requiring stratified splitting and balanced evaluation metrics (F1-score, Precision, Recall).
4. **Non-definitive Warning**: Classification is probabilistic based on lexical and morphological patterns. Classification as spam does not legally establish fraudulent intent.
