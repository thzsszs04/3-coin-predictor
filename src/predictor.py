from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_DATA_PATH = Path("/Users/macbookpro/Desktop/Thesis/data/final_merged_prices_fg_vol.csv")

COIN_OPTIONS = {
    "Bitcoin": {
        "label": "BTC",
        "price_col": "bitcoin_price",
        "volume_col": "BTCUSDT_volume",
        "return_col": "BTC_Return",
        "prefix": "BTC",
    },
    "Ethereum": {
        "label": "ETH",
        "price_col": "ethereum_price",
        "volume_col": "ETHUSDT_volume",
        "return_col": "ETH_Return",
        "prefix": "ETH",
    },
    "Solana": {
        "label": "SOL",
        "price_col": "solana_price",
        "volume_col": "SOLUSDT_volume",
        "return_col": "SOL_Return",
        "prefix": "SOL",
    },
}

MODEL_OPTIONS = {
    "SARIMAX": "sarimax",
    "XGBoost": "xgboost",
}

EXOG_COLS = ["FearGreed", "Market_Volatility_Monthly", "Market_Volume"]
LAGS = [1, 2, 3, 7]


def to_ms(dt_utc: datetime) -> int:
    return int(pd.Timestamp(dt_utc, tz="UTC").timestamp() * 1000)


def fetch_binance_klines(
    symbol: str,
    interval: str = "1d",
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    limit: int = 1000,
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    base = "https://data-api.binance.vision/api/v3/klines"
    if end_utc is None:
        end_utc = datetime.utcnow()
    if start_utc is None:
        start_utc = end_utc - timedelta(days=732)

    rows: list[dict[str, object]] = []
    cursor = to_ms(start_utc)
    end_ms = to_ms(end_utc)

    while True:
        response = requests.get(
            base,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": cursor,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        for candle in data:
            rows.append(
                {
                    "open_time": pd.to_datetime(candle[0], unit="ms", utc=True),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "close_time": pd.to_datetime(candle[6], unit="ms", utc=True),
                }
            )

        cursor = data[-1][6] + 1
        if cursor >= end_ms:
            break
        time.sleep(sleep_sec)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"Tidak ada data Binance untuk {symbol}.")
    return frame.drop_duplicates(subset=["open_time"]).set_index("open_time").sort_index()


def get_binance_data(symbol: str) -> pd.DataFrame:
    frame = fetch_binance_klines(symbol=symbol)
    frame["Date"] = frame.index.date
    frame[f"{symbol}_price"] = frame["close"]
    frame[f"{symbol}_volume"] = frame["volume"]
    return frame[["Date", f"{symbol}_price", f"{symbol}_volume"]]


def get_fear_greed(days: int = 730) -> pd.DataFrame:
    response = requests.get(f"https://api.alternative.me/fng/?limit={days}&format=json", timeout=30)
    response.raise_for_status()
    data = response.json()["data"]
    frame = pd.DataFrame(data)
    frame["Date"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="s").dt.date
    frame["FearGreed"] = frame["value"].astype(float)
    return frame[["Date", "FearGreed"]]


def fetch_live_raw_dataset() -> pd.DataFrame:
    btc = get_binance_data("BTCUSDT")
    eth = get_binance_data("ETHUSDT")
    sol = get_binance_data("SOLUSDT")

    for frame in (btc, eth, sol):
        frame["Date"] = pd.to_datetime(frame["Date"])

    merged = (
        btc.merge(eth, on="Date", how="outer")
        .merge(sol, on="Date", how="outer")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    fear_greed = get_fear_greed(730)
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()
    fear_greed["Date"] = pd.to_datetime(fear_greed["Date"]).dt.normalize()

    frame = merged.rename(
        columns={
            "BTCUSDT_price": "bitcoin_price",
            "ETHUSDT_price": "ethereum_price",
            "SOLUSDT_price": "solana_price",
        }
    )
    frame = frame.merge(fear_greed, on="Date", how="left")
    frame = frame.sort_values("Date").reset_index(drop=True)

    frame[["bitcoin_price", "ethereum_price", "solana_price"]] = frame[
        ["bitcoin_price", "ethereum_price", "solana_price"]
    ].ffill()
    frame["FearGreed"] = frame["FearGreed"].ffill()

    frame["BTC_Return"] = frame["bitcoin_price"].pct_change()
    frame["ETH_Return"] = frame["ethereum_price"].pct_change()
    frame["SOL_Return"] = frame["solana_price"].pct_change()

    scale = np.sqrt(30)
    frame["BTC_Volatility_Monthly"] = frame["BTC_Return"].rolling(30).std() * scale
    frame["ETH_Volatility_Monthly"] = frame["ETH_Return"].rolling(30).std() * scale
    frame["SOL_Volatility_Monthly"] = frame["SOL_Return"].rolling(30).std() * scale
    frame["Market_Volatility_Monthly"] = frame[
        ["BTC_Volatility_Monthly", "ETH_Volatility_Monthly", "SOL_Volatility_Monthly"]
    ].mean(axis=1)

    return frame


def read_local_raw_dataset(data_path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def load_dataset(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    try:
        frame = fetch_live_raw_dataset()
        source = "live"
    except Exception:
        frame = read_local_raw_dataset(data_path)
        source = "local"

    frame = frame.sort_values("Date").reset_index(drop=True)

    frame = frame.dropna(
        subset=[
            "BTC_Volatility_Monthly",
            "ETH_Volatility_Monthly",
            "SOL_Volatility_Monthly",
        ]
    ).reset_index(drop=True)
    frame = frame.ffill().bfill()

    frame["Market_Volume"] = (
        frame["BTCUSDT_volume"] +
        frame["ETHUSDT_volume"] +
        frame["SOLUSDT_volume"]
    )

    frame["BTC_Return"] = frame["bitcoin_price"].pct_change()
    frame["ETH_Return"] = frame["ethereum_price"].pct_change()
    frame["SOL_Return"] = frame["solana_price"].pct_change()

    for window in [7, 14, 30]:
        frame[f"BTC_MA_{window}"] = frame["bitcoin_price"].rolling(window).mean()
        frame[f"ETH_MA_{window}"] = frame["ethereum_price"].rolling(window).mean()
        frame[f"SOL_MA_{window}"] = frame["solana_price"].rolling(window).mean()

    for window in [7, 14, 30]:
        frame[f"BTC_Vol_{window}"] = frame["BTC_Return"].rolling(window).std()
        frame[f"ETH_Vol_{window}"] = frame["ETH_Return"].rolling(window).std()
        frame[f"SOL_Vol_{window}"] = frame["SOL_Return"].rolling(window).std()

    frame["FearGreed_norm"] = (
        (frame["FearGreed"] - frame["FearGreed"].min()) /
        (frame["FearGreed"].max() - frame["FearGreed"].min())
    )

    for lag in LAGS:
        frame[f"BTC_Lag_{lag}"] = frame["bitcoin_price"].shift(lag)
        frame[f"ETH_Lag_{lag}"] = frame["ethereum_price"].shift(lag)
        frame[f"SOL_Lag_{lag}"] = frame["solana_price"].shift(lag)

        frame[f"BTC_Return_Lag_{lag}"] = frame["BTC_Return"].shift(lag)
        frame[f"ETH_Return_Lag_{lag}"] = frame["ETH_Return"].shift(lag)
        frame[f"SOL_Return_Lag_{lag}"] = frame["SOL_Return"].shift(lag)

    frame = frame.dropna().reset_index(drop=True)
    frame.attrs["source"] = source
    return frame


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100


def cumulative_return(predictions: np.ndarray) -> float:
    return ((predictions[-1] - predictions[0]) / predictions[0]) * 100


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(calculate_mape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Directional Accuracy": float(directional_accuracy(y_true, y_pred)),
        "Cumulative Return": float(cumulative_return(y_pred)),
    }


def build_sarimax_residual_diagnostics(residuals: pd.Series, max_lag: int = 20) -> dict[str, object]:
    clean_residuals = pd.Series(residuals).replace([np.inf, -np.inf], np.nan).dropna()
    residual_std = clean_residuals.std()
    if pd.notna(residual_std) and residual_std > 0:
        clean_residuals = clean_residuals[abs(clean_residuals - clean_residuals.mean()) <= 6 * residual_std]
    if clean_residuals.empty:
        return {
            "residuals": pd.DataFrame(columns=["Residual"]),
            "autocorrelation": pd.DataFrame(columns=["Lag", "Autocorrelation"]),
            "ljung_box": {"lag": 0, "statistic": np.nan, "p_value": np.nan, "interpretation": "Unavailable"},
        }

    available_lag = int(min(max_lag, max(1, len(clean_residuals) - 1)))
    autocorrelation = pd.DataFrame(
        {
            "Lag": list(range(1, available_lag + 1)),
            "Autocorrelation": [float(clean_residuals.autocorr(lag=lag)) for lag in range(1, available_lag + 1)],
        }
    )

    lb_lag = int(min(10, available_lag))
    ljung_box = acorr_ljungbox(clean_residuals, lags=[lb_lag], return_df=True)
    p_value = float(ljung_box["lb_pvalue"].iloc[0])
    statistic = float(ljung_box["lb_stat"].iloc[0])
    interpretation = "Pass" if p_value >= 0.05 else "Pattern detected"

    return {
        "residuals": pd.DataFrame({"Residual": clean_residuals.to_numpy()}),
        "autocorrelation": autocorrelation,
        "ljung_box": {
            "lag": lb_lag,
            "statistic": statistic,
            "p_value": p_value,
            "interpretation": interpretation,
        },
    }


def create_future_exog_trend(df: pd.DataFrame, exog_cols: list[str], days: int = 7) -> pd.DataFrame:
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    recent = df[exog_cols].tail(14).copy()

    future_values: dict[str, np.ndarray] = {}
    for col in exog_cols:
        y_recent = recent[col].values
        x_recent = np.arange(len(y_recent))
        slope, intercept = np.polyfit(x_recent, y_recent, 1)
        x_future = np.arange(len(y_recent), len(y_recent) + days)
        future_values[col] = intercept + slope * x_future

    return pd.DataFrame(future_values, index=future_dates)


def create_future_external_features(df: pd.DataFrame, external_cols: list[str], days: int = 7) -> pd.DataFrame:
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    recent = df[external_cols].tail(14).copy()

    future_values: dict[str, np.ndarray] = {}
    for col in external_cols:
        y_recent = recent[col].values
        x_recent = np.arange(len(y_recent))
        slope, intercept = np.polyfit(x_recent, y_recent, 1)
        x_future = np.arange(len(y_recent), len(y_recent) + days)
        future_values[col] = intercept + slope * x_future

    return pd.DataFrame(future_values, index=future_dates)


def build_sarimax_base(df: pd.DataFrame) -> pd.DataFrame:
    sarimax_df = df.copy()
    sarimax_df = sarimax_df.set_index("Date").sort_index()
    sarimax_df["log_btc"] = np.log(sarimax_df["bitcoin_price"])
    sarimax_df["log_eth"] = np.log(sarimax_df["ethereum_price"])
    sarimax_df["log_sol"] = np.log(sarimax_df["solana_price"])
    return sarimax_df.dropna()


def run_sarimax_prediction(coin_name: str, base_df: pd.DataFrame, days: int = 7) -> dict[str, object]:
    config = COIN_OPTIONS[coin_name]
    df = build_sarimax_base(base_df)
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    log_col = f"log_{config['label'].lower()}"
    y_train = train[log_col]
    y_test = test[log_col]

    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(train[EXOG_COLS]),
        index=train.index,
        columns=EXOG_COLS,
    )
    x_test = pd.DataFrame(
        scaler.transform(test[EXOG_COLS]),
        index=test.index,
        columns=EXOG_COLS,
    )

    auto_model = auto_arima(
        y_train,
        X=x_train,
        seasonal=False,
        start_p=0,
        max_p=4,
        start_q=0,
        max_q=4,
        d=None,
        max_d=2,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    order = auto_model.order

    fitted_model = SARIMAX(
        y_train,
        exog=x_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    residual_diagnostics = build_sarimax_residual_diagnostics(fitted_model.resid)

    rolling_preds_log: list[float] = []
    current_model = fitted_model
    for index in range(len(test)):
        next_exog = x_test.iloc[[index]]
        forecast_log = current_model.forecast(steps=1, exog=next_exog)
        rolling_preds_log.append(float(forecast_log.iloc[0]))
        actual_log_value = y_test.iloc[index]
        current_model = current_model.append(endog=[actual_log_value], exog=next_exog, refit=False)

    predicted_prices = np.exp(np.array(rolling_preds_log))
    actual_prices = test[config["price_col"]].values
    metrics = build_metrics(actual_prices, predicted_prices)

    future_exog_raw = create_future_exog_trend(df, EXOG_COLS, days=days)
    future_exog_scaled = pd.DataFrame(
        scaler.transform(future_exog_raw),
        index=future_exog_raw.index,
        columns=EXOG_COLS,
    )
    forecast_log = current_model.forecast(steps=days, exog=future_exog_scaled)
    forecast_prices = np.exp(np.array(forecast_log))

    test_predictions = pd.DataFrame(
        {
            "Date": test.index,
            "Actual Price": actual_prices,
            "Predicted Price": predicted_prices,
        }
    )

    recent_actual = df[config["price_col"]].tail(30).reset_index()
    recent_actual = recent_actual.rename(columns={config["price_col"]: "Historical Price"})
    forecast_table = pd.DataFrame(
        {
            "Date": future_exog_scaled.index,
            "Coin": config["label"],
            "Predicted Price": np.round(forecast_prices, 2),
        }
    )

    forecast_chart = pd.DataFrame(
        {
            "Date": list(recent_actual["Date"]) + list(forecast_table["Date"]),
            "Historical Price": list(recent_actual["Historical Price"]) + [np.nan] * len(forecast_table),
            "Predicted Price": [np.nan] * len(recent_actual) + list(forecast_table["Predicted Price"]),
        }
    )

    return {
        "coin_name": coin_name,
        "model_name": "SARIMAX",
        "data_source": base_df.attrs.get("source", "unknown"),
        "data_range": (df.index[0], df.index[-1]),
        "metrics": metrics,
        "test_predictions": test_predictions,
        "test_period": (test.index[0], test.index[-1]),
        "forecast_table": forecast_table,
        "forecast_chart": forecast_chart,
        "forecast_days": days,
        "order": order,
        "residual_diagnostics": residual_diagnostics,
    }


def build_xgboost_base(df: pd.DataFrame) -> pd.DataFrame:
    xgb_df = df.copy()
    xgb_df = xgb_df.set_index("Date").sort_index()
    xgb_df["BTC_Return"] = xgb_df["bitcoin_price"].pct_change()
    xgb_df["ETH_Return"] = xgb_df["ethereum_price"].pct_change()
    xgb_df["SOL_Return"] = xgb_df["solana_price"].pct_change()

    for lag in LAGS:
        xgb_df[f"BTC_Lag_{lag}"] = xgb_df["bitcoin_price"].shift(lag)
        xgb_df[f"ETH_Lag_{lag}"] = xgb_df["ethereum_price"].shift(lag)
        xgb_df[f"SOL_Lag_{lag}"] = xgb_df["solana_price"].shift(lag)

        xgb_df[f"BTC_Return_Lag_{lag}"] = xgb_df["BTC_Return"].shift(lag)
        xgb_df[f"ETH_Return_Lag_{lag}"] = xgb_df["ETH_Return"].shift(lag)
        xgb_df[f"SOL_Return_Lag_{lag}"] = xgb_df["SOL_Return"].shift(lag)

    return xgb_df.dropna()


def get_xgb_features(prefix: str) -> list[str]:
    return [
        f"{prefix}_Lag_1",
        f"{prefix}_Lag_2",
        f"{prefix}_Lag_3",
        f"{prefix}_Lag_7",
        f"{prefix}_Return_Lag_1",
        f"{prefix}_Return_Lag_2",
        f"{prefix}_Return_Lag_3",
        f"{prefix}_Return_Lag_7",
        "FearGreed",
        "Market_Volatility_Monthly",
        "Market_Volume",
        f"{prefix}USDT_volume",
    ]


def build_xgb_model() -> XGBRegressor:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is not available on this machine yet. "
            "On macOS, install the OpenMP runtime with `brew install libomp`, "
            "then reinstall xgboost if needed."
        ) from exc

    return XGBRegressor(
        n_estimators=650,
        learning_rate=0.045,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def run_xgboost_prediction(coin_name: str, base_df: pd.DataFrame, days: int = 7) -> dict[str, object]:
    config = COIN_OPTIONS[coin_name]
    df = build_xgboost_base(base_df)
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    features = get_xgb_features(config["prefix"])
    target_col = config["return_col"]

    x_train = train[features]
    x_test = test[features]
    y_train = train[target_col]
    y_test_price = test[config["price_col"]]

    model = build_xgb_model()
    model.fit(x_train, y_train)
    predicted_returns = model.predict(x_test)

    previous_actual_prices = pd.concat(
        [train[config["price_col"]].iloc[[-1]], test[config["price_col"]].iloc[:-1]]
    ).values
    predicted_prices = previous_actual_prices * (1 + predicted_returns)
    actual_prices = y_test_price.values
    metrics = build_metrics(actual_prices, predicted_prices)
    feature_importance = (
        pd.DataFrame(
            {
                "Feature": features,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    external_cols = [
        "FearGreed",
        "Market_Volatility_Monthly",
        "Market_Volume",
        config["volume_col"],
    ]
    future_external = create_future_external_features(df, external_cols, days=days)
    price_history = list(df[config["price_col"]].values)
    return_history = list(df[target_col].values)

    predicted_prices_7: list[float] = []
    predicted_returns_7: list[float] = []
    for current_date in future_external.index:
        feature_row = {
            f"{config['prefix']}_Lag_1": price_history[-1],
            f"{config['prefix']}_Lag_2": price_history[-2],
            f"{config['prefix']}_Lag_3": price_history[-3],
            f"{config['prefix']}_Lag_7": price_history[-7],
            f"{config['prefix']}_Return_Lag_1": return_history[-1],
            f"{config['prefix']}_Return_Lag_2": return_history[-2],
            f"{config['prefix']}_Return_Lag_3": return_history[-3],
            f"{config['prefix']}_Return_Lag_7": return_history[-7],
            "FearGreed": future_external.loc[current_date, "FearGreed"],
            "Market_Volatility_Monthly": future_external.loc[current_date, "Market_Volatility_Monthly"],
            "Market_Volume": future_external.loc[current_date, "Market_Volume"],
            config["volume_col"]: future_external.loc[current_date, config["volume_col"]],
        }
        x_future = pd.DataFrame([feature_row], columns=features)
        pred_return = float(model.predict(x_future)[0])
        pred_price = price_history[-1] * (1 + pred_return)
        predicted_returns_7.append(pred_return)
        predicted_prices_7.append(pred_price)
        price_history.append(pred_price)
        return_history.append(pred_return)

    test_predictions = pd.DataFrame(
        {
            "Date": y_test_price.index,
            "Actual Price": actual_prices,
            "Predicted Price": predicted_prices,
        }
    )

    recent_actual = df[config["price_col"]].tail(30).reset_index()
    recent_actual = recent_actual.rename(columns={config["price_col"]: "Historical Price"})
    forecast_table = pd.DataFrame(
        {
            "Date": future_external.index,
            "Coin": config["label"],
            "Predicted Price": np.round(predicted_prices_7, 2),
            "Predicted Return (%)": np.round(np.array(predicted_returns_7) * 100, 4),
        }
    )

    forecast_chart = pd.DataFrame(
        {
            "Date": list(recent_actual["Date"]) + list(forecast_table["Date"]),
            "Historical Price": list(recent_actual["Historical Price"]) + [np.nan] * len(forecast_table),
            "Predicted Price": [np.nan] * len(recent_actual) + list(forecast_table["Predicted Price"]),
        }
    )

    return {
        "coin_name": coin_name,
        "model_name": "XGBoost",
        "data_source": base_df.attrs.get("source", "unknown"),
        "data_range": (df.index[0], df.index[-1]),
        "metrics": metrics,
        "test_predictions": test_predictions,
        "test_period": (test.index[0], test.index[-1]),
        "feature_importance": feature_importance,
        "forecast_table": forecast_table,
        "forecast_chart": forecast_chart,
        "forecast_days": days,
    }


def run_prediction(
    coin_name: str,
    model_name: str,
    data_path: Path | str = DEFAULT_DATA_PATH,
    days: int = 7,
) -> dict[str, object]:
    df = load_dataset(data_path)
    model_key = MODEL_OPTIONS[model_name]

    if model_key == "sarimax":
        return run_sarimax_prediction(coin_name, df, days=days)
    if model_key == "xgboost":
        return run_xgboost_prediction(coin_name, df, days=days)
    raise ValueError(f"Unsupported model: {model_name}")


def create_future_exog_average(df: pd.DataFrame, exog_cols: list[str], days: int = 7) -> pd.DataFrame:
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    recent_exog_mean = df[exog_cols].tail(7).mean()
    return pd.DataFrame(
        [recent_exog_mean.values] * days,
        index=future_dates,
        columns=exog_cols,
    )


def calculate_confidence(metrics: dict[str, float]) -> tuple[float, str]:
    directional = float(metrics.get("Directional Accuracy", 0))
    normalized_r2 = float(np.clip(metrics.get("R2", 0), 0, 1) * 100)
    inverse_mape = float(np.clip(100 - metrics.get("MAPE", 100), 0, 100))
    confidence = (directional * 0.4) + (normalized_r2 * 0.3) + (inverse_mape * 0.3)
    if confidence >= 80:
        level = "High"
    elif confidence >= 60:
        level = "Medium"
    else:
        level = "Low"
    return float(confidence), level


def classify_backtest_trends(table: pd.DataFrame, last_actual_price: float) -> list[str]:
    labels: list[str] = []
    previous_actual = last_actual_price
    previous_predicted = last_actual_price
    sideways_threshold = 0.002

    for row in table.itertuples(index=False):
        actual_price = float(getattr(row, "Actual_Price"))
        predicted_price = float(getattr(row, "Predicted_Price"))
        actual_return = (actual_price - previous_actual) / previous_actual
        predicted_return = (predicted_price - previous_predicted) / previous_predicted

        if abs(actual_return) < sideways_threshold and abs(predicted_return) < sideways_threshold:
            labels.append("🟡 Sideways")
        elif np.sign(actual_return) == np.sign(predicted_return):
            labels.append("🟢 Correct Bullish" if actual_return > 0 else "🟢 Correct Bearish")
        else:
            labels.append("🔴 Wrong Direction")

        previous_actual = actual_price
        previous_predicted = predicted_price

    return labels


def finalize_backtest_result(
    *,
    coin_name: str,
    model_name: str,
    data_source: str,
    data_range: tuple[pd.Timestamp, pd.Timestamp],
    cutoff_date: pd.Timestamp,
    historical_actual: pd.Series,
    comparison_table: pd.DataFrame,
) -> dict[str, object]:
    comparison_table = comparison_table.dropna(subset=["Actual Price", "Predicted Price"]).copy()
    if comparison_table.empty:
        raise ValueError("No actual future data is available for the selected backtesting period.")

    comparison_table["Error"] = comparison_table["Actual Price"] - comparison_table["Predicted Price"]
    comparison_table["Absolute Error"] = comparison_table["Error"].abs()
    comparison_table["APE (%)"] = (comparison_table["Absolute Error"] / comparison_table["Actual Price"]) * 100

    metrics = build_metrics(
        comparison_table["Actual Price"].to_numpy(),
        comparison_table["Predicted Price"].to_numpy(),
    )
    confidence_score, confidence_level = calculate_confidence(metrics)

    trend_input = comparison_table.rename(columns={"Actual Price": "Actual_Price", "Predicted Price": "Predicted_Price"})
    comparison_table["Trend"] = classify_backtest_trends(trend_input, float(historical_actual.iloc[-1]))

    table_display = comparison_table.copy()
    numeric_cols = ["Actual Price", "Predicted Price", "Error", "Absolute Error", "APE (%)"]
    table_display[numeric_cols] = table_display[numeric_cols].round(2)

    chart_data = pd.DataFrame(
        {
            "Date": list(historical_actual.index) + list(comparison_table["Date"]),
            "Historical Price": list(historical_actual.values) + [np.nan] * len(comparison_table),
            "Actual Future Price": [np.nan] * len(historical_actual) + list(comparison_table["Actual Price"]),
            "Predicted Price": [np.nan] * len(historical_actual) + list(comparison_table["Predicted Price"]),
        }
    )

    return {
        "coin_name": coin_name,
        "coin_label": COIN_OPTIONS[coin_name]["label"],
        "model_name": model_name,
        "data_source": data_source,
        "data_range": data_range,
        "cutoff_date": cutoff_date,
        "backtest_period": (comparison_table["Date"].iloc[0], comparison_table["Date"].iloc[-1]),
        "metrics": metrics,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "forecast_days": len(comparison_table),
        "comparison_table": table_display,
        "chart_data": chart_data,
    }


def run_sarimax_backtest(
    coin_name: str,
    base_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    days: int = 7,
) -> dict[str, object]:
    config = COIN_OPTIONS[coin_name]
    df = build_sarimax_base(base_df)
    cutoff_date = pd.to_datetime(cutoff_date)
    cutoff_date = min(cutoff_date, df.index[-days - 1])

    train_df = df[df.index <= cutoff_date].copy()
    actual_future = df[
        (df.index > cutoff_date) &
        (df.index <= cutoff_date + pd.Timedelta(days=days))
    ][config["price_col"]]

    log_col = f"log_{config['label'].lower()}"
    y_train = train_df[log_col]
    x_train_raw = train_df[EXOG_COLS]

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train_raw),
        index=x_train_raw.index,
        columns=EXOG_COLS,
    )
    auto_model = auto_arima(
        y_train,
        X=x_train_scaled,
        seasonal=False,
        start_p=0,
        max_p=4,
        start_q=0,
        max_q=4,
        d=None,
        max_d=2,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    fitted_model = SARIMAX(
        y_train,
        exog=x_train_scaled,
        order=auto_model.order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    future_exog_raw = create_future_exog_average(train_df, EXOG_COLS, days=days)
    future_exog_scaled = pd.DataFrame(
        scaler.transform(future_exog_raw),
        index=future_exog_raw.index,
        columns=EXOG_COLS,
    )
    forecast_log = fitted_model.forecast(steps=days, exog=future_exog_scaled)
    forecast_price = np.exp(forecast_log)

    comparison_table = pd.DataFrame(
        {
            "Date": forecast_price.index,
            "Coin": config["label"],
            "Predicted Price": forecast_price.values,
            "Actual Price": actual_future.reindex(forecast_price.index).values,
        }
    )

    return finalize_backtest_result(
        coin_name=coin_name,
        model_name="SARIMAX",
        data_source=base_df.attrs.get("source", "unknown"),
        data_range=(df.index[0], df.index[-1]),
        cutoff_date=cutoff_date,
        historical_actual=train_df[config["price_col"]].tail(30),
        comparison_table=comparison_table,
    )


def build_backtest_xgb_model():
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is not available on this machine yet. "
            "On macOS, install the OpenMP runtime with `brew install libomp`, "
            "then reinstall xgboost if needed."
        ) from exc

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def run_xgboost_backtest(
    coin_name: str,
    base_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    days: int = 7,
) -> dict[str, object]:
    config = COIN_OPTIONS[coin_name]
    df = build_xgboost_base(base_df)
    cutoff_date = pd.to_datetime(cutoff_date)
    cutoff_date = min(cutoff_date, df.index[-days - 1])

    features = get_xgb_features(config["prefix"])
    target_col = config["return_col"]
    volume_col = config["volume_col"]
    train_df = df[df.index <= cutoff_date].copy()
    actual_future = df[
        (df.index > cutoff_date) &
        (df.index <= cutoff_date + pd.Timedelta(days=days))
    ][config["price_col"]]

    model = build_backtest_xgb_model()
    model.fit(train_df[features], train_df[target_col])

    external_cols = ["FearGreed", "Market_Volatility_Monthly", "Market_Volume", volume_col]
    future_external = create_future_external_features(train_df, external_cols, days=days)
    price_history = list(train_df[config["price_col"]].values)
    return_history = list(train_df[target_col].values)

    predicted_prices: list[float] = []
    predicted_returns: list[float] = []
    for current_date in future_external.index:
        feature_row = {
            f"{config['prefix']}_Lag_1": price_history[-1],
            f"{config['prefix']}_Lag_2": price_history[-2],
            f"{config['prefix']}_Lag_3": price_history[-3],
            f"{config['prefix']}_Lag_7": price_history[-7],
            f"{config['prefix']}_Return_Lag_1": return_history[-1],
            f"{config['prefix']}_Return_Lag_2": return_history[-2],
            f"{config['prefix']}_Return_Lag_3": return_history[-3],
            f"{config['prefix']}_Return_Lag_7": return_history[-7],
            "FearGreed": future_external.loc[current_date, "FearGreed"],
            "Market_Volatility_Monthly": future_external.loc[current_date, "Market_Volatility_Monthly"],
            "Market_Volume": future_external.loc[current_date, "Market_Volume"],
            volume_col: future_external.loc[current_date, volume_col],
        }
        x_future = pd.DataFrame([feature_row], columns=features)
        pred_return = float(model.predict(x_future)[0])
        pred_price = price_history[-1] * (1 + pred_return)
        predicted_returns.append(pred_return)
        predicted_prices.append(pred_price)
        price_history.append(pred_price)
        return_history.append(pred_return)

    comparison_table = pd.DataFrame(
        {
            "Date": future_external.index,
            "Coin": config["label"],
            "Predicted Price": predicted_prices,
            "Predicted Return (%)": np.array(predicted_returns) * 100,
            "Actual Price": actual_future.reindex(future_external.index).values,
        }
    )

    return finalize_backtest_result(
        coin_name=coin_name,
        model_name="XGBoost",
        data_source=base_df.attrs.get("source", "unknown"),
        data_range=(df.index[0], df.index[-1]),
        cutoff_date=cutoff_date,
        historical_actual=train_df[config["price_col"]].tail(30),
        comparison_table=comparison_table,
    )


def run_backtesting(
    coin_name: str,
    model_name: str,
    cutoff_date: pd.Timestamp,
    data_path: Path | str = DEFAULT_DATA_PATH,
    days: int = 7,
) -> dict[str, object]:
    df = load_dataset(data_path)
    model_key = MODEL_OPTIONS[model_name]
    if model_key == "sarimax":
        return run_sarimax_backtest(coin_name, df, cutoff_date=cutoff_date, days=days)
    if model_key == "xgboost":
        return run_xgboost_backtest(coin_name, df, cutoff_date=cutoff_date, days=days)
    raise ValueError(f"Unsupported model: {model_name}")
