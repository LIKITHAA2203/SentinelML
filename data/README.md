# SentinelML Dataset Guide

## 1. Primary Dataset: European Credit Card Fraud Detection
The project uses the widely cited European Credit Card Fraud dataset (collected by the Machine Learning Group at ULB - Université Libre de Bruxelles).

- **Source**: Kaggle Credit Card Fraud Detection Dataset ([Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) or OpenML (Dataset ID 42175).
- **License**: Database Contents License (DbCL) / Open Data Commons Open Database License (ODbL) v1.0.
- **File Name**: `creditcard.csv` (approx. 143.8 MB uncompressed).
- **Target Location**: Place the uncompressed file at:
  ```text
  data/raw/creditcard.csv
  ```

## 2. Dataset Properties & Statistics
- **Total Records**: 284,807 transactions.
- **Timeframe**: 2 days of European cardholder transactions (September 2013).
- **Class Imbalance**:
  - `Class = 0` (Legitimate): 284,315 (~99.828%)
  - `Class = 1` (Fraudulent): 492 (~0.172%)
- **Features (30 total input features)**:
  - `Time`: Elapsed seconds between this transaction and the first transaction in the dataset.
  - `Amount`: Transaction amount in Euros (€). Highly skewed (median around €22.00, max €25,691.16).
  - `V1` to `V28`: Principal Component Analysis (PCA) transformed numerical features to preserve privacy and confidentiality of cardholders.
  - `Class`: Response variable (0 = Legitimate, 1 = Fraudulent).

## 3. Safe Development / Quickstart Mode
If you are developing locally on a laptop without an active Kaggle account or want an immediate instant dry-run without waiting for 150 MB download:
Run our reproducible generator:
```powershell
python scripts/generate_demo_data.py --samples 5000 --out data/raw/creditcard.csv
```
This produces a fully compatible synthetic development dataset preserving the exact column schema, PCA variance properties, and realistic severe class imbalance (~0.17% fraud).
When you download the official Kaggle `creditcard.csv`, simply replace this file in `data/raw/creditcard.csv` and re-run training!
