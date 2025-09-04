import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Tokens to analyze
tokens = ["BTC-USD", "ETH-USD", "DOGE-USD"]

for token in tokens:
    print(f"\n=== Analyzing {token} ===")
    
    # Fetch data
    df = yf.download(token, start="2022-01-01", end="2024-12-31")
    df = df.reset_index()
    
    # Feature engineering
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df.dropna(inplace=True)
    
    # Target variable
    df["Trend"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    
    # Features and labels
    X = df[["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]]
    y = df["Trend"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n📊 {name} on {token}")
        print(f"Accuracy: {acc:.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print(f"\nModel Accuracies for {token}: {results}")
