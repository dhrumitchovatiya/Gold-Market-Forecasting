# Gold Market Forecasting | Aurum Regime Continuum

## Project Overview

This project explores whether daily gold price direction can be predicted using rule-based trading logic and machine learning models built only from OHLC (Open, High, Low, Close) price data.

The goal was not only to maximize prediction accuracy, but also to understand the limitations of candle-based forecasting in a highly macro-sensitive asset such as gold.

---

## Experiment Results Summary

| Experiment   | Model                      | Accuracy |
| ------------ | -------------------------- | -------- |
| Experiment 1 | Rule-Based Trading System  | 35.09%   |
| Experiment 2 | Baseline ML (XGBoost)      | 31.64%   |
| Experiment 3 | Wick-Enhanced ML (XGBoost) | 38.63%   |

## Dataset

* Asset: Gold
* Data Frequency: Daily OHLC
* Original Rows: 6358
* Cleaned Rows (2010+): 3822
* Removed Corrupted Rows: 201

Data cleaning included:

* Filtering data from 2010 onward
* Removing invalid rows where Open = High = Low = Close
* Sorting by time and removing missing values

---

## Experiment 1: Rule-Based Trading System

A handcrafted trading model using:

* ATR expansion
* Range expansion
* Trend filters
* Market regime classification

### Result

Accuracy: 35.09%

### Insight

Rigid trading rules struggled to adapt to changing gold market behavior.

---

## Experiment 2: Baseline Machine Learning

Model:

* XGBoost

Features:

* ATR14
* ATR expansion
* Range %
* Signed body
* Signed body ratio
* Weighted momentum
* WMA slope difference

### Result

Accuracy: 31.64%

### Insight

Basic candle-body and trend features alone provided weak predictive signal.

---

## Experiment 3: Wick-Enhanced Machine Learning

Model:

* XGBoost

Additional Features:

* Upper wick ratio
* Lower wick ratio
* Lag1
* Lag2
* Lag3
* Range expansion

### Result

Accuracy: 38.63%

### Insight

Adding wick rejection and previous-candle memory improved performance significantly, suggesting candle rejection patterns contain useful short-term information.

---

## Final Conclusions

### 1. Gold OHLC data contains limited directional signal

Even the best candle-based ML model achieved 38.63% accuracy, indicating that daily OHLC data alone provides limited predictive power for reliable direction forecasting.

### 2. Wick rejection improves short-term prediction

Upper and lower wick features improved model accuracy by nearly 7%, showing that candle rejection behavior carries meaningful short-term market information.

### 3. Gold is highly non-stationary

Gold price behavior changes significantly across market regimes and reacts strongly to macroeconomic events such as:

* interest rate changes
* inflation
* central bank policy
* geopolitical uncertainty

This makes purely technical forecasting challenging.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

