import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "gold_prices.csv"

START_INDEX = 100

# Step 1 thresholds
ATR_EXPANSION_THRESHOLD = 1.05
RANGE_EXPANSION_THRESHOLD = 1.05

# Step 2 thresholds
BODY_RATIO_STRONG = 0.70
BODY_RATIO_MEDIUM = 0.50

BODY_EXPANSION_STRONG = 1.30
BODY_EXPANSION_MEDIUM = 1.10


# =====================================================
# LOAD + CLEAN DATA
# =====================================================

def load_data(path):
    df = pd.read_csv(path)

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

    # Only 2010+
    df = df[df["date"] >= "2010-01-01"].copy()
    print("Rows After 2010 Filter:", len(df))

    # Remove bad OHLC rows
    bad_rows = (
        (df["open"] == df["high"]) &
        (df["high"] == df["low"]) &
        (df["low"] == df["close"])
    )

    print("Bad Rows Removed:", bad_rows.sum())

    df = df[~bad_rows].copy()

    print("Final Clean Rows:", len(df))

    return df


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(df):
    df = df.copy()

    # Candle body
    df["body_size"] = abs(
        df["close"] - df["open"]
    )

    # Candle range
    df["range_size"] = (
        df["high"] - df["low"]
    )

    df["range_size"] = df["range_size"].replace(
        0,
        np.nan
    )

    # Body conviction
    df["body_ratio"] = (
        df["body_size"] /
        df["range_size"]
    )

    # Candle direction
    df["bullish"] = np.where(
        df["close"] > df["open"],
        1,
        0
    )

    df["bearish"] = np.where(
        df["close"] < df["open"],
        1,
        0
    )

    # ATR (daily)
    df["atr"] = (
        df["range_size"]
        .rolling(14)
        .mean()
    )

    # Range rolling averages
    df["avg_range_7"] = (
        df["range_size"]
        .rolling(7)
        .mean()
    )

    df["avg_range_3"] = (
        df["range_size"]
        .rolling(3)
        .mean()
    )

    # ATR rolling averages
    df["avg_atr_7"] = (
        df["atr"]
        .rolling(7)
        .mean()
    )

    df["avg_atr_3"] = (
        df["atr"]
        .rolling(3)
        .mean()
    )

    # Body rolling average
    df["avg_body_7"] = (
        df["body_size"]
        .rolling(7)
        .mean()
    )

    # WMA
    df["wma20"] = WMA(df["close"], 20)
    df["wma50"] = WMA(df["close"], 50)

    df["wma20_slope"] = df["wma20"].diff()
    df["wma50_slope"] = df["wma50"].diff()

    df["slope_difference"] = (
        df["wma20_slope"] -
        df["wma50_slope"]
    )

    return df


# =====================================================
# WMA FUNCTION
# =====================================================

def WMA(series, period):
    weights = np.arange(1, period + 1)

    return series.rolling(period).apply(
        lambda prices:
        np.dot(prices, weights) / weights.sum(),
        raw=True
    )


# =====================================================
# ACTUAL LABEL
# =====================================================

def get_actual_label(day_row):
    if day_row["body_ratio"] < 0.30:
        return "NEUTRAL"

    if day_row["close"] > day_row["open"]:
        return "BULLISH"

    return "BEARISH"

# =====================================================
# CANDLE STRENGTH SCORE (for last 3 candles)
# =====================================================

def candle_strength(candle):
    score = 0

    body_ratio = candle["body_ratio"]
    body_size = candle["body_size"]
    avg_body = candle["avg_body_7"]

    if pd.isna(avg_body):
        return 0

    body_expansion = body_size / avg_body

    # Body ratio score
    if body_ratio >= BODY_RATIO_STRONG:
        score += 10
    elif body_ratio >= BODY_RATIO_MEDIUM:
        score += 5

    # Body expansion score
    if body_expansion >= BODY_EXPANSION_STRONG:
        score += 10
    elif body_expansion >= BODY_EXPANSION_MEDIUM:
        score += 5

    return score


# =====================================================
# STEP 1 — TRADABLE FILTER
# =====================================================

def is_tradable(history):
    latest = history.iloc[-1]

    atr_expansion = (
        latest["avg_atr_3"] /
        latest["avg_atr_7"]
    )

    range_expansion = (
        latest["avg_range_3"] /
        latest["avg_range_7"]
    )

    atr_ok = atr_expansion >= ATR_EXPANSION_THRESHOLD
    range_ok = range_expansion >= RANGE_EXPANSION_THRESHOLD

    return atr_ok and range_ok


# =====================================================
# STEP 2 — DIRECTION ENGINE
# =====================================================

def predict_direction(history):
    latest = history.iloc[-1]

    bull_score = 0
    bear_score = 0

    # ============================================
    # STEP 2A — 7 DAY MARKET PRESSURE (50)
    # ============================================

    slope = latest["slope_difference"]

    atr_expansion = (
        latest["avg_atr_3"] /
        latest["avg_atr_7"]
    )

    range_expansion = (
        latest["avg_range_3"] /
        latest["avg_range_7"]
    )

    pressure_score = 0

    # Trend pressure
    if abs(slope) > history["slope_difference"].abs().rolling(20).mean().iloc[-1]:
        pressure_score += 20
    else:
        pressure_score += 10

    # ATR pressure
    if atr_expansion >= 1.2:
        pressure_score += 15
    elif atr_expansion >= 1.05:
        pressure_score += 8

    # Range pressure
    if range_expansion >= 1.2:
        pressure_score += 15
    elif range_expansion >= 1.05:
        pressure_score += 8

    if slope > 0:
        bull_score += pressure_score
    else:
        bear_score += pressure_score

    # ============================================
    # STEP 2B — LAST 3 CANDLE PATTERN (50)
    # ============================================

    recent_3 = history.iloc[-3:]

    for _, candle in recent_3.iterrows():
        strength = candle_strength(candle)

        if candle["bullish"] == 1:
            bull_score += strength
        elif candle["bearish"] == 1:
            bear_score += strength

    # ============================================
    # FINAL DECISION
    # ============================================

    score_gap = bull_score - bear_score

    # Reversal logic
    if slope > 0 and score_gap <= -15:
        return "REVERSAL"

    if slope < 0 and score_gap >= 15:
        return "REVERSAL"

    if score_gap >= 20:
        return "BULLISH"

    if score_gap <= -20:
        return "BEARISH"

    return "NEUTRAL"


# =====================================================
# SINGLE DAY PREDICTION
# =====================================================

def predict_day(history):
    if len(history) < 60:
        return "NEUTRAL"

    if not is_tradable(history):
        return "NON-TRADABLE"

    return predict_direction(history)

# =====================================================
# WALK-FORWARD BACKTEST
# =====================================================

def run_backtest(df):
    predictions = []
    actuals = []

    tradable_count = 0
    non_tradable_count = 0

    print("\n====================================")
    print("RUNNING WALK-FORWARD BACKTEST")
    print("====================================\n")

    for i in range(START_INDEX, len(df)):

        history = df.iloc[:i].copy()
        today = df.iloc[i]

        prediction = predict_day(history)

        if prediction == "NON-TRADABLE":
            non_tradable_count += 1
            continue

        tradable_count += 1

        actual = get_actual_label(today)

        predictions.append(prediction)
        actuals.append(actual)

    results = pd.DataFrame({
        "prediction": predictions,
        "actual": actuals
    })

    return results, tradable_count, non_tradable_count


# =====================================================
# EVALUATION
# =====================================================

def evaluate_backtest(results, tradable_count, non_tradable_count):

    print("\n====================================")
    print("BACKTEST RESULTS")
    print("====================================\n")

    total_predictions = len(results)

    print(f"Tradable Days: {tradable_count}")
    print(f"Non-Tradable Days: {non_tradable_count}")
    print(f"Total Predictions: {total_predictions}")

    if total_predictions == 0:
        print("No predictions generated.")
        return

    correct = (
        results["prediction"] == results["actual"]
    ).sum()

    overall_accuracy = (
        correct / total_predictions
    ) * 100

    print("\nOVERALL ACCURACY")
    print(f"{overall_accuracy:.2f}%")

    print("\nCLASS-WISE ACCURACY\n")

    # Actual classes only (reversal is prediction-only)
    actual_classes = [
        "BULLISH",
        "BEARISH",
        "NEUTRAL"
    ]

    for cls in actual_classes:
        subset = results[
            results["actual"] == cls
        ]

        if len(subset) == 0:
            acc = 0
        else:
            acc = (
                (
                    subset["prediction"] == subset["actual"]
                ).sum()
                / len(subset)
            ) * 100

        print(f"{cls}: {acc:.2f}%")

    print("\n====================================")
    print("PREDICTION DISTRIBUTION")
    print("====================================\n")
    print(results["prediction"].value_counts())

    print("\n====================================")
    print("ACTUAL DISTRIBUTION")
    print("====================================\n")
    print(results["actual"].value_counts())

    print("\n====================================")
    print("CONFUSION TABLE")
    print("====================================\n")

    confusion = pd.crosstab(
        results["actual"],
        results["prediction"]
    )

    print(confusion)


# =====================================================
# LIVE PREDICTION
# =====================================================

def live_prediction(df):
    print("\n====================================")
    print("LIVE PREDICTION")
    print("====================================\n")

    # Use past data only
    history = df.iloc[:-1].copy()

    prediction = predict_day(history)

    print("Predicted Today Direction:")
    print(prediction)


# =====================================================
# MAIN
# =====================================================

def main():
    df = load_data(DATA_PATH)

    df = create_features(df)

    df = df.dropna().reset_index(drop=True)

    results, tradable_count, non_tradable_count = run_backtest(df)

    evaluate_backtest(
        results,
        tradable_count,
        non_tradable_count
    )

    live_prediction(df)


if __name__ == "__main__":
    main()