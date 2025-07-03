import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

autism_screening_adult = fetch_ucirepo(id=426)
X = autism_screening_adult.data.features.copy()
y = autism_screening_adult.data.targets.copy()

print(X.columns)
features = [f'A{i}_Score' for i in range(1, 11)] + ['age', 'gender']
X = X[features]

X['gender'] = X['gender'].map({'m': 1, 'f': 0})
X['age'] = pd.to_numeric(X['age'], errors='coerce')
for col in [f'A{i}_Score' for i in range(1, 11)]:
    if X[col].dtype == 'O':
        X[col] = X[col].map({'yes': 1, 'no': 0}).fillna(X[col])
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

print("Colunas alvo disponíveis:", y.columns)
target_col = y.columns[0]
print("Coluna alvo escolhida:", target_col)
y = y[target_col].map({'YES': 1, 'NO': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
