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

    df.columns = [col.lower() for col in df.columns]

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("\n====================================")
    print("DATA CLEANING")
    print("====================================")

    print("Original Rows:", len(df))

    df = df[df["date"] >= START_DATE].copy()

    print("Rows After 2010 Filter:", len(df))

    bad_rows = (
        (df["open"] == df["high"]) &
        (df["high"] == df["low"]) &
        (df["low"] == df["close"])
    )

    print("Bad Rows Removed:", bad_rows.sum())

    df = df[~bad_rows].copy()
    df = df.reset_index(drop=True)

    print("Final Clean Rows:", len(df))

    return df


# ============================================
# BASIC FEATURES
# ============================================

def create_basic_features(df):
    df = df.copy()

    df["range_size"] = (
        df["high"] - df["low"]
    )

    df["range_size"] = df["range_size"].replace(
        0,
        np.nan
    )

    df["range_percent"] = (
        df["range_size"] /
        df["close"]
    ) * 100

    df["signed_body"] = (
        df["close"] - df["open"]
    )

    df["signed_body_ratio"] = (
        df["signed_body"] /
        df["range_size"]
    )

    # Wick features
    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    df["wick_ratio"] = (
        (df["upper_wick"] + df["lower_wick"]) /
        df["range_size"]
    )

    print("\n====================================")
    print("PART 1 CHECK")
    print("====================================")

    print(df[[
        "date",
        "signed_body_ratio",
        "upper_wick",
        "lower_wick",
        "wick_ratio"
    ]].tail())

    return df

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

    # --------------------------------
    # ATR FEATURES
    # --------------------------------
    df["atr14"] = (
        df["range_size"]
        .rolling(14)
        .mean()
    )

    df["atr_expansion"] = (
        df["atr14"] /
        df["atr14"].rolling(10).mean()
    )

    # --------------------------------
    # RANGE EXPANSION
    # --------------------------------
    df["range_expansion"] = (
        df["range_size"] /
        df["range_size"].rolling(5).mean()
    )

    # --------------------------------
    # LAG RETURNS
    # IMPORTANT:
    # Previous trading candles, not calendar days
    # --------------------------------
    df["daily_return"] = (
        (
            df["close"] -
            df["close"].shift(1)
        )
        /
        df["close"].shift(1)
    ) * 100

    df["lag1"] = df["daily_return"].shift(1)
    df["lag2"] = df["daily_return"].shift(2)
    df["lag3"] = df["daily_return"].shift(3)

    # --------------------------------
    # WICK RATIOS
    # --------------------------------
    df["upper_wick_ratio"] = (
        df["upper_wick"] /
        df["range_size"]
    )

    df["lower_wick_ratio"] = (
        df["lower_wick"] /
        df["range_size"]
    )

    # --------------------------------
    # WMA TREND
    # --------------------------------
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
        "range_expansion",
        "lag1",
        "lag2",
        "lag3",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "slope_difference"
    ]].tail())

# ============================================
# LABEL CREATION (3-DAY HORIZON)
# ============================================

def create_labels(df):
    df = df.copy()

    df["future_return_3d"] = (
        (
            df["close"].shift(-3) -
            df["close"]
        )
        /
        df["close"]
    ) * 100

    conditions = [
        df["future_return_3d"] <= -0.75,

        (
            (df["future_return_3d"] > -0.75)
            &
            (df["future_return_3d"] < 0.75)
        ),

        df["future_return_3d"] >= 0.75
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

    # Remove rolling + future NaNs
    df = df.dropna().reset_index(drop=True)

    # ====================================
    # FINAL FEATURE SET
    # ====================================

    features = [
        "atr14",
        "atr_expansion",
        "range_percent",
        "range_expansion",
        "signed_body",
        "signed_body_ratio",
        "lag1",
        "lag2",
        "lag3",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "slope_difference"
    ]

    X = df[features]
    y = df["target"]

    # Time-series split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\n====================================")
    print("PART 3 CHECK")
    print("====================================")

    print("Final Rows:", len(df))
    print("Train Size:", len(X_train))
    print("Test Size:", len(X_test))

    print("\nTARGET DISTRIBUTION")
    print(y.value_counts().sort_index())

    print("\nCLASS MEANING")
    print("0 = BEARISH")
    print("1 = NEUTRAL")
    print("2 = BULLISH")

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# MAIN
# ============================================

def main():
    df = load_data()
    df = create_basic_features(df)
    df = create_advanced_features(df)
    df = create_labels(df)

    df = df.dropna().reset_index(drop=True)

    features = [
        "atr14",
        "atr_expansion",
        "range_percent",
        "range_expansion",
        "signed_body",
        "signed_body_ratio",
        "lag1",
        "lag2",
        "lag3",
        "upper_wick_ratio",
        "lower_wick_ratio",
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

    # ============================================
    # XGBOOST
    # ============================================

    model = XGBClassifier(
        n_estimators=500,
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

    # ============================================
    # REPORT
    # ============================================

    print("\n====================================")
    print("XGBOOST V5 RESULTS")
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

    # ============================================
    # FEATURE IMPORTANCE
    # ============================================

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

    # ============================================
    # FINAL ACCURACY
    # ============================================

    accuracy = (
        (y_pred == y_test).mean()
    ) * 100

    print("\n====================================")
    print("FINAL ACCURACY")
    print("====================================\n")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # ============================================
    # LATEST HISTORICAL PREDICTION
    # ============================================

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