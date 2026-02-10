from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Protocol, Optional

import numpy as np
import pandas as pd


class Model(Protocol):
    name: str

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
          {
            "forecast": pd.DataFrame with columns ["date","yhat","yhat_lower","yhat_upper"],
            "metrics": dict (optional)
          }
        """


def _infer_target_series(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column.")
    if "balance" in df.columns and df["balance"].notna().any():
        y = df["balance"].astype(float)
        y.name = "balance"
        return y
    if "net_flow" in df.columns and df["net_flow"].notna().any():
        # If balance isn't available, forecast cumulative net flow (what the UI plots).
        y = df["net_flow"].astype(float).cumsum()
        y.name = "cumulative_net_flow"
        return y
    raise ValueError("Dataset must contain 'balance' or 'net_flow'.")


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    last_date = pd.to_datetime(last_date)
    start = last_date + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=int(horizon), freq="D")


def _simple_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    err = (y_true - y_pred).to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


@dataclass(frozen=True)
class MovingAverageModel:
    name: str = "A0 • Moving Average"

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        window = int(params.get("window") or 14)
        if window < 1:
            window = 1

        y = _infer_target_series(df)
        if len(y) < 2:
            raise ValueError("Not enough data to forecast.")

        tail = y.tail(window)
        mu = float(tail.mean())
        sigma = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0

        dates = _future_dates(df["date"].iloc[-1], horizon)
        yhat = np.full(len(dates), mu, dtype=float)
        ci = 1.96 * sigma
        fc = pd.DataFrame(
            {
                "date": dates,
                "yhat": yhat,
                "yhat_lower": yhat - ci,
                "yhat_upper": yhat + ci,
            }
        )

        # Backtest last N points with one-step MA to provide a sanity metric.
        n = min(30, max(0, len(y) - 1))
        metrics: Dict[str, float] = {}
        if n >= 5:
            preds = []
            trues = []
            y_arr = y.to_numpy(dtype=float)
            for i in range(len(y_arr) - n, len(y_arr)):
                start = max(0, i - window)
                hist = y_arr[start:i]
                if hist.size == 0:
                    continue
                preds.append(float(np.mean(hist)))
                trues.append(float(y_arr[i]))
            if len(preds) >= 5:
                metrics = _simple_metrics(pd.Series(trues), pd.Series(preds))

        return {"forecast": fc, "metrics": metrics}


@dataclass(frozen=True)
class ArimaModel:
    name: str = "A1 • ARIMA"

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except Exception as e:  # pragma: no cover
            raise ImportError("statsmodels is required for ARIMA forecasts.") from e

        p = int(params.get("p") or 1)
        d = int(params.get("d") or 1)
        q = int(params.get("q") or 1)
        s = int(params.get("s") or 7)
        if p < 0 or d < 0 or q < 0:
            raise ValueError("ARIMA p,d,q must be non-negative integers.")
        if s < 1:
            s = 7

        y = _infer_target_series(df).astype(float)
        if len(y) < max(20, (p + d + q + 1)):
            raise ValueError("Not enough data to fit ARIMA reliably.")

        # Basic weekly seasonality by default; harmless if signal isn't seasonal.
        model = SARIMAX(
            y,
            order=(p, d, q),
            seasonal_order=(0, 0, 0, s),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

        fc_res = res.get_forecast(steps=int(horizon))
        mean = fc_res.predicted_mean
        ci = fc_res.conf_int(alpha=0.05)

        dates = _future_dates(df["date"].iloc[-1], horizon)
        fc = pd.DataFrame(
            {
                "date": dates,
                "yhat": mean.to_numpy(dtype=float),
                "yhat_lower": ci.iloc[:, 0].to_numpy(dtype=float),
                "yhat_upper": ci.iloc[:, 1].to_numpy(dtype=float),
            }
        )

        metrics: Dict[str, float] = {}
        try:
            fitted = pd.Series(res.fittedvalues, index=y.index).dropna()
            aligned = y.loc[fitted.index]
            n = min(30, len(fitted))
            if n >= 5:
                metrics = _simple_metrics(aligned.tail(n), fitted.tail(n))
        except Exception:
            metrics = {}

        return {"forecast": fc, "metrics": metrics}


@dataclass(frozen=True)
class NaiveLastModel:
    """Baseline: forecast last observed value."""
    name: str = "A2 • Naive Last"

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        y = _infer_target_series(df).astype(float)
        if len(y) < 1:
            raise ValueError("Not enough data to forecast.")

        last_val = float(y.iloc[-1])
        dates = _future_dates(df["date"].iloc[-1], horizon)
        fc = pd.DataFrame(
            {
                "date": dates,
                "yhat": np.full(len(dates), last_val, dtype=float),
                "yhat_lower": np.full(len(dates), np.nan, dtype=float),
                "yhat_upper": np.full(len(dates), np.nan, dtype=float),
            }
        )
        return {"forecast": fc, "metrics": {}}


@dataclass(frozen=True)
class WeekdayMeanModel:
    """Forecast based on weekday pattern (useful for treasury flows)."""
    name: str = "A3 • Weekday Mean"

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        y = _infer_target_series(df).astype(float)
        if len(y) < 7:
            raise ValueError("Not enough data for weekday pattern (need at least 7 days).")

        # Group by weekday (0=Monday, 6=Sunday)
        df_with_dow = df.copy()
        df_with_dow["date"] = pd.to_datetime(df_with_dow["date"])
        df_with_dow["dow"] = df_with_dow["date"].dt.dayofweek
        y_series = pd.Series(y.values, index=df_with_dow["dow"].values)
        weekday_means = y_series.groupby(level=0).mean().to_dict()

        dates = _future_dates(df["date"].iloc[-1], horizon)
        dates_pd = pd.to_datetime(dates)
        dow_future = dates_pd.dayofweek
        yhat = np.array([weekday_means.get(d, y.mean()) for d in dow_future], dtype=float)

        fc = pd.DataFrame(
            {
                "date": dates,
                "yhat": yhat,
                "yhat_lower": np.full(len(dates), np.nan, dtype=float),
                "yhat_upper": np.full(len(dates), np.nan, dtype=float),
            }
        )

        # Simple backtest metric
        metrics: Dict[str, float] = {}
        if len(y) >= 14:
            preds = []
            trues = []
            for i in range(max(0, len(y) - 14), len(y)):
                if i < len(df_with_dow):
                    dow = df_with_dow["dow"].iloc[i]
                    preds.append(weekday_means.get(dow, y.mean()))
                    trues.append(float(y.iloc[i]))
            if len(preds) >= 5:
                metrics = _simple_metrics(pd.Series(trues), pd.Series(preds))

        return {"forecast": fc, "metrics": metrics}


@dataclass(frozen=True)
class ETSModel:
    """Exponential Smoothing (Holt-Winters)."""
    name: str = "A4 • Exponential Smoothing"

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except Exception as e:
            raise ImportError("statsmodels is required for ETS forecasts.") from e

        y = _infer_target_series(df).astype(float)
        if len(y) < 10:
            raise ValueError("Not enough data for ETS (need at least 10 points).")

        seasonal_period = int(params.get("seasonal_period") or 7)
        if seasonal_period < 2:
            seasonal_period = 7

        dates = _future_dates(df["date"].iloc[-1], horizon)

        try:
            # Try seasonal model first
            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add" if len(y) >= 2 * seasonal_period else None,
                seasonal_periods=seasonal_period if len(y) >= 2 * seasonal_period else None,
                initialization_method="estimated",
            )
            res = model.fit(optimized=True, use_brute=False)
            fc_res = res.forecast(steps=int(horizon))
            mean = fc_res.values

            # Approximate CI from residuals
            fitted = res.fittedvalues.dropna()
            if len(fitted) > 0:
                resid = (y.loc[fitted.index] - fitted).abs()
                sigma = float(resid.std()) if len(resid) > 1 else 0.0
                ci_width = 1.96 * sigma
            else:
                ci_width = 0.0

            fc = pd.DataFrame(
                {
                    "date": dates,
                    "yhat": mean.astype(float),
                    "yhat_lower": (mean - ci_width).astype(float),
                    "yhat_upper": (mean + ci_width).astype(float),
                }
            )

            metrics: Dict[str, float] = {}
            if len(fitted) >= 5:
                aligned = y.loc[fitted.index]
                n = min(30, len(fitted))
                metrics = _simple_metrics(aligned.tail(n), fitted.tail(n))

            return {"forecast": fc, "metrics": metrics}
        except Exception:
            # Fallback to simple mean
            mu = float(y.mean())
            fc = pd.DataFrame(
                {
                    "date": dates,
                    "yhat": np.full(len(dates), mu, dtype=float),
                    "yhat_lower": np.full(len(dates), np.nan, dtype=float),
                    "yhat_upper": np.full(len(dates), np.nan, dtype=float),
                }
            )
            return {"forecast": fc, "metrics": {}}


@dataclass(frozen=True)
class RandomForestModel:
    """Random Forest regression with lag features."""
    name: str = "B0 • Random Forest"

    def predict(self, df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as e:
            raise ImportError("scikit-learn is required for Random Forest forecasts.") from e

        y = _infer_target_series(df).astype(float)
        if len(y) < 20:
            raise ValueError("Not enough data for Random Forest (need at least 20 points).")

        # Build simple lag features
        n_lags = int(params.get("n_lags") or 7)
        n_lags = min(n_lags, max(1, len(y) // 4))

        X_list = []
        y_list = []
        dates_list = []

        for i in range(n_lags, len(y)):
            X_list.append([float(y.iloc[i - j - 1]) for j in range(n_lags)])
            y_list.append(float(y.iloc[i]))
            dates_list.append(df["date"].iloc[i])

        if len(X_list) < 5:
            raise ValueError("Not enough data after feature construction.")

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators") or 100),
            max_depth=int(params.get("max_depth") or 10) if params.get("max_depth") else None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Forecast recursively
        dates = _future_dates(df["date"].iloc[-1], horizon)
        yhat = []
        last_window = [float(y.iloc[-j - 1]) for j in range(n_lags)]

        for _ in range(int(horizon)):
            pred = float(model.predict([last_window])[0])
            yhat.append(pred)
            last_window = [pred] + last_window[:-1]

        # Approximate CI from training residuals
        y_pred_train = model.predict(X_train)
        resid = np.abs(y_train - y_pred_train)
        sigma = float(np.std(resid)) if len(resid) > 1 else 0.0
        ci_width = 1.96 * sigma

        fc = pd.DataFrame(
            {
                "date": dates,
                "yhat": np.array(yhat, dtype=float),
                "yhat_lower": (np.array(yhat) - ci_width).astype(float),
                "yhat_upper": (np.array(yhat) + ci_width).astype(float),
            }
        )

        metrics: Dict[str, float] = {}
        if len(y_train) >= 5:
            metrics = _simple_metrics(pd.Series(y_train), pd.Series(y_pred_train))

        return {"forecast": fc, "metrics": metrics}


_MODELS: List[Model] = [
    MovingAverageModel(),
    ArimaModel(),
    NaiveLastModel(),
    WeekdayMeanModel(),
    ETSModel(),
    RandomForestModel(),
]
_BY_NAME: Dict[str, Model] = {m.name: m for m in _MODELS}


def list_models() -> List[str]:
    return [m.name for m in _MODELS]


def get(name: str) -> Model:
    m: Optional[Model] = _BY_NAME.get(name)
    if m is None:
        raise KeyError(f"Unknown model: {name}")
    return m

