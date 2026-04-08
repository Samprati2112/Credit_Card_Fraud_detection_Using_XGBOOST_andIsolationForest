# Credit Card Fraud Detection (Final Notebook)

This project detects fraudulent credit card transactions using an anomaly-augmented machine learning pipeline built in Jupyter Notebook.

Primary implementation file:
- `creditCard_Fraud_Detection_final.ipynb`

## Project Summary

The final notebook builds a fraud detection system using:
- Feature engineering (`hour`, `log_amount`)
- Data scaling with `StandardScaler`
- `IsolationForest` for anomaly scoring
- `XGBoost` classifier trained on augmented features
- Threshold tuning using F-beta (`beta = 1.5`) from Precision-Recall
- Full evaluation with classification report, confusion matrix, ROC-AUC, and PR-AUC
- Custom transaction simulation for final decision testing

## Dataset

The notebook currently reads:

```python
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\ML_project\creditcard_fraud_detection.csv")
```

If running on another machine, update this path to a relative path, for example:

```python
df = pd.read_csv("creditcard_fraud_detection.csv")
```

## 💾 Dataset

Due to GitHub's file size limits, the 143MB dataset is not hosted in this repository. You can download the exact dataset used for this project directly from Kaggle:
- [Download the European Credit Card Fraud Dataset Here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**To run this code locally:**
1. Download the dataset from the link above.
2. Extract the `creditcard.csv` file.
3. Place it in a folder named `data/` in the root directory of this project.

## Repository Structure

```text
ML_project/
|--Base_Paper
|-- creditCard_Fraud_Detection_final.ipynb
|-- credit_card_fraud_dataset.csv
|-- Final_Report
`-- README.md
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

## Run The Project

1. Open the notebook `creditCard_Fraud_Detection_final.ipynb` in VS Code or Jupyter.
2. Run cells from top to bottom.
3. Review outputs:
- Classification report
- Core metrics (Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1)
- Confusion matrix plot
- ROC and Precision-Recall curves
- Base paper comparison table
- Custom transaction evaluation result

## Model Pipeline

1. Load and preprocess data.
2. Create engineered features.
3. Train/test split and scaling.
4. Train Isolation Forest and generate anomaly scores.
5. Concatenate anomaly score with scaled features.
6. Train XGBoost on augmented data.
7. Tune decision threshold using F-beta on validation set.
8. Evaluate on test set and visualize results.

## Notes

- This is an imbalanced classification problem; metrics like PR-AUC and Recall are important alongside Accuracy.
- The notebook is the source of truth for the final model workflow.
- For reproducibility and deployment, moving code into Python modules is recommended.

## Future Improvements

- Add `requirements.txt` with pinned versions.
- Use relative paths everywhere.
- Export the trained model (`joblib`/`pickle`) for inference scripts or API deployment.
- Add k-fold cross-validation and experiment tracking.

## Author

ML Project for Credit Card Fraud Detection.
