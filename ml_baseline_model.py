import pandas as pd
import numpy as np

# ============================================
# CONFIG
# ============================================

DATA_PATH = "gold_prices.csv"
START_DATE = "2010-01-01"

# ============================================
# LOAD DATA
# ============================================

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Lowercase column names
    df.columns = [col.lower() for col in df.columns]

    # Parse date safely
    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["date"])

    # Sort
    df = df.sort_values("date").reset_index(drop=True)

    print("\n====================================")
    print("DATA CLEANING")
    print("====================================")

    print("Original Rows:", len(df))

    # Keep only 2010+
    df = df[df["date"] >= START_DATE].copy()

    print("Rows After 2010 Filter:", len(df))

    # Remove corrupted OHLC rows
    bad_rows = (
        (df["open"] == df["high"]) &
        (df["high"] == df["low"]) &
        (df["low"] == df["close"])
    )

    print("Bad Rows Removed:", bad_rows.sum())

    df = df[~bad_rows].copy()

    print("Final Clean Rows:", len(df))
    
    df = df.reset_index(drop=True)
    return df


# ============================================
# BASIC FEATURES
# ============================================

def create_basic_features(df):
    df = df.copy()

    # Total candle range
    df["range_size"] = (
        df["high"] - df["low"]
    )

    # Avoid divide by zero
    df["range_size"] = df["range_size"].replace(
        0,
        np.nan
    )

    # Volatility feature
    df["range_percent"] = (
        df["range_size"] /
        df["close"]
    ) * 100

    # Directional candle body
    df["signed_body"] = (
        df["close"] - df["open"]
    )

    # IMPORTANT FEATURE
    # Direction + momentum + conviction
    df["signed_body_ratio"] = (
        df["signed_body"] /
        df["range_size"]
    )

    return df


# ============================================
# MAIN
# ============================================

def main():
    df = load_data()
    df = create_basic_features(df)

    print("\n====================================")
    print("PART 1 CHECK")
    print("====================================")

    print(df[[
        "date",
        "open",
        "high",
        "low",
        "close",
        "range_percent",
        "signed_body",
        "signed_body_ratio"
    ]].tail())


if __name__ == "__main__":
    main()

# ============================================
# WMA FUNCTION
# ============================================

def WMA(series, period):
    weights = np.arange(1, period + 1)

    return series.rolling(period).apply(
        lambda prices:
        np.dot(prices, weights) / weights.sum(),
        raw=True
    )


# ============================================
# ADVANCED FEATURES
# ============================================

def create_advanced_features(df):
    df = df.copy()

    # ----------------------------
    # ATR FEATURES
    # ----------------------------
    df["atr14"] = (
        df["range_size"]
        .rolling(14)
        .mean()
    )

    df["atr_expansion"] = (
        df["atr14"] /
        df["atr14"].rolling(10).mean()
    )

    # ----------------------------
    # DAILY RETURN (%)
    # ----------------------------
    df["daily_return"] = (
        (
            df["close"] -
            df["close"].shift(1)
        )
        /
        df["close"].shift(1)
    ) * 100

    # ----------------------------
    # WEIGHTED MOMENTUM
    # Recent days weighted higher
    # (5,4,3,2,1)
    # ----------------------------
    df["weighted_momentum"] = (
        5 * df["daily_return"].shift(1) +
        4 * df["daily_return"].shift(2) +
        3 * df["daily_return"].shift(3) +
        2 * df["daily_return"].shift(4) +
        1 * df["daily_return"].shift(5)
    ) / 15

    # ----------------------------
    # WMA TREND
    # ----------------------------
    df["wma20"] = WMA(
        df["close"],
        20
    )

    df["wma50"] = WMA(
        df["close"],
        50
    )

    df["wma20_slope"] = (
        df["wma20"] -
        df["wma20"].shift(1)
    )

    df["wma50_slope"] = (
        df["wma50"] -
        df["wma50"].shift(1)
    )

    df["slope_difference"] = (
        df["wma20_slope"] -
        df["wma50_slope"]
    )

    return df


# ============================================
# MAIN
# ============================================

def main():
    df = load_data()
    df = create_basic_features(df)
    df = create_advanced_features(df)

    print("\n====================================")
    print("PART 2 CHECK")
    print("====================================")

    print(df[[
        "date",
        "atr14",
        "atr_expansion",
        "daily_return",
        "weighted_momentum",
        "wma20",
        "wma50",
        "slope_difference"
    ]].tail())


if __name__ == "__main__":
    main()    

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# LABEL CREATION
# Predict 5-day future move using only past data
# ============================================

def create_labels(df):
    df = df.copy()

    df["future_return_5d"] = (
        (
            df["close"].shift(-5) -
            df["close"]
        )
        /
        df["close"]
    ) * 100

    conditions = [
        df["future_return_5d"] <= -0.75,

        (
            (df["future_return_5d"] > -0.75)
            &
            (df["future_return_5d"] < 0.75)
        ),

        df["future_return_5d"] >= 0.75
    ]

    labels = [0, 1, 2]

    df["target"] = np.select(
        conditions,
        labels,
        default=1
    )

    return df


# ============================================
# MAIN
# ============================================

def main():
    df = load_data()
    df = create_basic_features(df)
    df = create_advanced_features(df)
    df = create_labels(df)

    # Remove rolling NaN rows
    df = df.dropna().reset_index(drop=True)

    # ========================================
    # FEATURE SET (NO LIQUIDITY)
    # ========================================

    features = [
        "atr14",
        "atr_expansion",
        "range_percent",
        "signed_body",
        "signed_body_ratio",
        "weighted_momentum",
        "slope_difference"
    ]

    X = df[features]
    y = df["target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\n====================================")
    print("MODEL DATA")
    print("====================================")
    print("Train Size:", len(X_train))
    print("Test Size:", len(X_test))

    print("\nTarget Distribution")
    print(y.value_counts().sort_index())

    # ========================================
    # XGBOOST
    # ========================================

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ========================================
    # REPORT
    # ========================================

    print("\n====================================")
    print("XGBOOST RESULTS")
    print("====================================\n")

    print(classification_report(
        y_test,
        y_pred
    ))

    print("\nCONFUSION MATRIX\n")
    print(confusion_matrix(
        y_test,
        y_pred
    ))

    # ========================================
    # FEATURE IMPORTANCE
    # ========================================

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE\n")
    print(importance)

    # ========================================
    # ACCURACY
    # ========================================

    accuracy = (
        (y_pred == y_test).mean()
    ) * 100

    print("\n====================================")
    print("FINAL ACCURACY")
    print("====================================\n")

    print(f"Overall Accuracy: {accuracy:.2f}%")

    # Latest historical prediction
    latest = X_test.iloc[-1:]
    pred = model.predict(latest)[0]
    probs = model.predict_proba(latest)[0]

    labels_map = {
        0: "BEARISH",
        1: "NEUTRAL",
        2: "BULLISH"
    }

    print("\n====================================")
    print("LATEST HISTORICAL ANALYSIS")
    print("====================================\n")

    print("Prediction:", labels_map[pred])

    print("\nCONFIDENCE")
    for i, prob in enumerate(probs):
        print(f"{labels_map[i]}: {prob*100:.2f}%")


if __name__ == "__main__":
    main()