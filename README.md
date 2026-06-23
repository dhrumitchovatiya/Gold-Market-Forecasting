# Gold Market Forecasting (Aurum Regime Continuum)

## Project Overview

This project explores whether daily gold price direction can be predicted using rule-based trading logic and machine learning models built only from OHLC (Open, High, Low, Close) price data.

The goal was not only to maximize prediction accuracy, but also to understand the limitations of candle-based forecasting in a highly macro-sensitive asset such as gold.

---

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

### 1. Gold OHLC alone has limited predictive power

Even the best candle-based ML model plateaued below 40% accuracy.

### 2. Wick rejection improves short-term prediction

Upper and lower wick behavior carried meaningful information about market rejection.

### 3. Gold is highly non-stationary

Gold reacts strongly to external macroeconomic conditions such as:

* Interest rates
* Inflation
* Central bank policy
* Geopolitical shocks

This makes purely technical prediction difficult.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
