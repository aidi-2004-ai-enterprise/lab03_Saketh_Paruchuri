"""
train.py:
- Loads seaborn penguins
- One‑hots sex + island
- Label‑encodes species
- Stratified split (80/20)
- Trains XGBoost (max_depth=3, n_estimators=100)
- Prints F1 scores
- Saves model JSON to app/data/model.json
"""

import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

def main():
    df = sns.load_dataset("penguins").dropna()
    X = df.drop("species", axis=1)
    y = df["species"]

    # Encode y
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # One-hot X[['sex','island']]
    Xo = pd.get_dummies(X, columns=["sex", "island"])

    # Save encoder classes
    os.makedirs("app/data", exist_ok=True)
    pd.Series(le.classes_).to_csv("app/data/target_classes.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        Xo, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    model = xgb.XGBClassifier(max_depth=3, n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)

    # Metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("Train F1:", f1_score(y_train, y_train_pred, average="macro"))
    print("Test F1:", f1_score(y_test, y_test_pred, average="macro"))
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # Save model JSON
    model.save_model("app/data/model.json")

if __name__ == "__main__":
    main()
