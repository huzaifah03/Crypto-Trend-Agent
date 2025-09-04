# Crypto-Trend-Agent
Machine Learning agent for cryptocurrency price trend prediction (BTC, ETH, DOGE).
# Cryptocurrency Price Trend Prediction Agent

## Objective
To build a machine learning agent that predicts cryptocurrency token price trends (BTC, ETH, DOGE) using historical data.

## Motivation
Crypto markets are volatile. Predicting short-term trends can assist traders in decision-making.

## Methodology
- Data Source: Yahoo Finance (BTC-USD, ETH-USD, DOGE-USD)
- Features: Open, High, Low, Close, Volume, MA5, MA10, Returns
- Models: Logistic Regression, Decision Tree, Random Forest
- Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report

## Results
- Random Forest achieved the best overall accuracy across all tokens.
- Logistic Regression struggled with non-linear patterns.
- Decision Tree performed moderately but prone to overfitting.

## Future Work
- Use advanced models (LSTMs, Gradient Boosting).
- Include sentiment and on-chain data.
- Deploy as a web or mobile app.

## How to Run
```bash
pip install -r requirements.txt
python crypto_trend_agent.py
