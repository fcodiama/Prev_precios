#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG (AQUÍ PONES LA MEDIA DE MAÑANA Y RUTAS)
# ============================================================
HIST_DIR = "precios"
HIST_SEP = ","

TOMORROW_DATE = "2025-12-23"
TOMORROW_MEAN_EUR_MWH = 140.55  # <-- AQUÍ

DAYS = 7

OUT_FINAL = "forecast_week_hourly_FINAL.csv"
OUT_SCEN_COLS = "forecast_week_hourly_SCENARIOS.csv"
OUT_SCEN_LONG = "forecast_week_hourly_SCENARIOS_long.csv"

EVAL_CSV = None
EVAL_SEP = ","
EVAL_OUT_PREFIX = "mae_"

LOW_FACTOR = 0.80
HIGH_FACTOR = 1.20

LOW_Q = 0.05
ALPHA_CLIP_Q = (0.05, 0.95)
LAMBDAS_7D = [1.00, 0.80, 0.65, 0.50, 0.40, 0.30, 0.25]

VALLEY_HOURS = {0, 1, 2, 3}
VALLEY_MULT_BASE = 0.55

REGIME_ULTRA_LOW = 60.0
REGIME_LOW = 90.0
REGIME_FLATTEN_W = {"ULTRA_LOW": 0.70, "LOW": 0.35, "NORMAL": 0.00}

FESTIVOS_FIJOS_ES = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

# ============================================================
# CALENDARIO / CLASES
# ============================================================
def day_type(dow: int, holiday: bool) -> str:
    if holiday or dow == 6:
        return "DOM"
    if dow == 5:
        return "SAT"
    return "LAB"

def season(month: int) -> str:
    if month in (12, 1, 2): return "INV"
    if month in (3, 4, 5):  return "PRI"
    if month in (6, 7, 8):  return "VER"
    return "OTO"

def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df["dow"] = df["date"].dt.weekday
    df["holiday"] = [(m, d) in FESTIVOS_FIJOS_ES for m, d in zip(df["month"], df["day"])]
    df["type"] = [day_type(dw, hol) for dw, hol in zip(df["dow"], df["holiday"])]
    df["season"] = [season(m) for m in df["month"]]
    df["class"] = df["type"] + "_" + df["season"]
    return df

def build_future_calendar(start_date: str, days: int) -> pd.DataFrame:
    start = pd.to_datetime(start_date).normalize()
    rows = []
    for i in range(days):
        d = start + pd.Timedelta(days=i)
        for h in range(24):
            rows.append({
                "date": d,
                "year": d.year,
                "month": d.month,
                "day": d.day,
                "hour": h
            })
    cal = pd.DataFrame(rows)
    cal = add_calendar_columns(cal)
    return cal[["date", "hour", "year", "month", "day", "class", "type"]]

# ============================================================
# CARGA HISTÓRICO
# ============================================================
def load_history_csvs(data_dir: str, sep: str = ",") -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("20*.csv"))
    if not files:
        raise FileNotFoundError(f"No encuentro CSVs 20*.csv en: {data_dir.resolve()}")

    df = pd.concat((pd.read_csv(f, sep=sep) for f in files), ignore_index=True)

    df = df.rename(columns={
        "Year": "year", "Month": "month", "Day": "day",
        "Hour": "hour_124", "Value": "price_eur_mwh"
    })
    missing = {"year", "month", "day", "hour_124", "price_eur_mwh"} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en histórico: {missing}. Esperado: Year,Month,Day,Hour,Value")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df["hour_124"] = pd.to_numeric(df["hour_124"], errors="coerce").astype("Int64")
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    df = df.dropna(subset=["year", "month", "day", "hour_124", "price_eur_mwh"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour_124"] = df["hour_124"].astype(int)

    # 1-24 -> 0-23
    df["hour"] = df["hour_124"] - 1
    if df["hour"].min() < 0 or df["hour"].max() > 23:
        raise ValueError("Horas fuera de 0-23 tras convertir Hour 1-24.")

    df = add_calendar_columns(df)
    df = df.dropna(subset=["price_eur_mwh"]).reset_index(drop=True)
    return df

# ============================================================
# SHAPE α + ESTADÍSTICAS
# ============================================================
def build_alpha_profiles(df_hist: pd.DataFrame, q=(0.05, 0.95)) -> pd.DataFrame:
    df = df_hist.copy()
    daily_mean = df.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day")
    df = df.merge(daily_mean, on="date", how="left")
    df["ratio"] = df["price_eur_mwh"] / df["p_mean_day"]

    lo_q, hi_q = q

    def clip_group(g: pd.DataFrame) -> pd.DataFrame:
        lo = g["ratio"].quantile(lo_q)
        hi = g["ratio"].quantile(hi_q)
        g["ratio"] = g["ratio"].clip(lo, hi)
        return g

    df = df.groupby(["class", "hour"], group_keys=False).apply(clip_group)

    alpha = (df.groupby(["class", "hour"])["ratio"]
               .mean()
               .reset_index()
               .rename(columns={"ratio": "alpha"}))
    return alpha

def build_class_daily_stats(df_hist: pd.DataFrame) -> pd.DataFrame:
    daily = (df_hist.groupby(["date", "class"])["price_eur_mwh"]
                    .mean()
                    .reset_index()
                    .rename(columns={"price_eur_mwh": "p_mean_day"}))
    stats = (daily.groupby("class")["p_mean_day"]
                  .agg(mu="mean", sigma="std", n="count")
                  .reset_index())
    return stats

def build_class_daily_low_quantiles(df_hist: pd.DataFrame, q=0.05) -> pd.DataFrame:
    daily = (df_hist.groupby(["date", "class"])["price_eur_mwh"]
                    .mean()
                    .reset_index()
                    .rename(columns={"price_eur_mwh": "p_mean_day"}))
    qlow = (daily.groupby("class")["p_mean_day"]
                  .quantile(q)
                  .rename("p_low")
                  .reset_index())
    return qlow

def calibrate_type_factor(df_hist: pd.DataFrame) -> dict:
    daily = (df_hist.groupby(["date", "type"])["price_eur_mwh"]
                    .mean()
                    .reset_index()
                    .rename(columns={"price_eur_mwh": "p_mean_day"}))
    base = daily.loc[daily["type"] == "LAB", "p_mean_day"].mean()
    out = {}
    for t in ["LAB", "SAT", "DOM"]:
        m = daily.loc[daily["type"] == t, "p_mean_day"].mean()
        out[t] = float(m / base) if base and base > 0 else 1.0
    return out

# ============================================================
# VALLE + RÉGIMEN
# ============================================================
def infer_regime(daily_mean: float) -> str:
    if daily_mean < REGIME_ULTRA_LOW:
        return "ULTRA_LOW"
    if daily_mean < REGIME_LOW:
        return "LOW"
    return "NORMAL"

def valley_multiplier(hour: int, k_anchor: float) -> float:
    if hour not in VALLEY_HOURS:
        return 1.0
    k = max(1.0, float(k_anchor))
    return 1.0 - (1.0 - VALLEY_MULT_BASE) / k

def apply_adjustments_one_day(day_df: pd.DataFrame) -> pd.DataFrame:
    df = day_df.copy()
    k = float(df["k_anchor"].iloc[0]) if "k_anchor" in df.columns else 1.0
    regime = df["regime"].iloc[0] if "regime" in df.columns else "NORMAL"
    w_flat = float(REGIME_FLATTEN_W.get(regime, 0.0))

    df["m_valley"] = [valley_multiplier(int(h), k) for h in df["hour"]]
    df["alpha_adj"] = df["alpha"] * df["m_valley"]

    if w_flat > 0:
        df["alpha_adj"] = (1.0 - w_flat) * df["alpha_adj"] + w_flat * 1.0

    mean_adj = float(df["alpha_adj"].mean())
    if mean_adj <= 0 or not np.isfinite(mean_adj):
        mean_adj = 1.0
    df["alpha_adj"] = df["alpha_adj"] / mean_adj

    df["Price_EUR_MWh"] = df["p_mean_pred_eur_mwh"] * df["alpha_adj"]
    return df

# ============================================================
# FORECAST
# ============================================================
def forecast_week_from_tomorrow_mean(df_hist: pd.DataFrame,
                                     tomorrow_date: str,
                                     tomorrow_mean_eur_mwh: float,
                                     days: int = 7,
                                     low_q: float = 0.05) -> pd.DataFrame:
    if days > len(LAMBDAS_7D):
        raise ValueError("DAYS > 7: amplía LAMBDAS_7D si quieres más de 7 días.")

    # sanity
    required = {"date", "hour", "price_eur_mwh", "class", "type"}
    miss = required - set(df_hist.columns)
    if miss:
        raise ValueError(f"df_hist sin columnas necesarias: {miss}. Tiene: {list(df_hist.columns)}")

    alpha = build_alpha_profiles(df_hist, q=ALPHA_CLIP_Q)
    stats = build_class_daily_stats(df_hist).set_index("class")
    mu_global = float(stats["mu"].mean())

    qlow_df = build_class_daily_low_quantiles(df_hist, q=low_q).set_index("class")
    p_low_global = float(qlow_df["p_low"].median()) if len(qlow_df) else 0.0

    type_factor = calibrate_type_factor(df_hist)

    # calendario futuro con horas
    cal = build_future_calendar(tomorrow_date, days)
    cal_days = cal.drop_duplicates(subset=["date"])[["date", "class", "type"]].reset_index(drop=True)

    # ancla
    start = pd.to_datetime(tomorrow_date).normalize()
    anchor_class = cal_days.loc[cal_days["date"] == start, "class"].iloc[0]
    mu_anchor = float(stats.loc[anchor_class, "mu"]) if anchor_class in stats.index else mu_global
    k_anchor = float(tomorrow_mean_eur_mwh / mu_anchor) if mu_anchor > 0 else 1.0

    # medias diarias previstas
    daily_pred = []
    for i, row in cal_days.iterrows():
        c, t = row["class"], row["type"]
        mu_c = float(stats.loc[c, "mu"]) if c in stats.index else mu_global

        lam = LAMBDAS_7D[i]
        kd = k_anchor ** lam
        ft = float(type_factor.get(t, 1.0))

        p_pred = mu_c * kd * ft

        p_low_c = float(qlow_df.loc[c, "p_low"]) if c in qlow_df.index else p_low_global
        p_pred = max(float(p_pred), float(p_low_c))

        daily_pred.append({"date": row["date"], "p_mean_pred_eur_mwh": float(p_pred), "regime": infer_regime(float(p_pred))})

    daily_fc = pd.DataFrame(daily_pred)
    daily_fc["k_anchor"] = k_anchor
    daily_fc["mu_anchor"] = mu_anchor
    daily_fc["TomorrowMean_EUR_MWh"] = float(tomorrow_mean_eur_mwh)
    daily_fc["LowQuantileUsed"] = float(low_q)
    daily_fc["ValleyHours"] = str(sorted(list(VALLEY_HOURS)))
    daily_fc["ValleyMultBase"] = float(VALLEY_MULT_BASE)

    # une cal horario con diarios y alpha
    out = cal.merge(daily_fc, on="date", how="left")
    out = out.merge(alpha, on=["class", "hour"], how="left")

    if out["alpha"].isna().any():
        missing = out.loc[out["alpha"].isna(), "class"].unique()
        raise ValueError(f"Faltan alphas para clases: {missing}")

    out = (out.sort_values(["date", "hour"])
              .groupby("date", group_keys=False)
              .apply(apply_adjustments_one_day)
              .reset_index(drop=True))

    out["Source"] = "FORECAST_MED"

    return out[[
        "date", "hour", "class", "type", "regime",
        "p_mean_pred_eur_mwh", "Price_EUR_MWh",
        "k_anchor", "mu_anchor",
        "TomorrowMean_EUR_MWh", "LowQuantileUsed",
        "ValleyHours", "ValleyMultBase",
        "Source"
    ]]

# ============================================================
# ESCENARIOS
# ============================================================
def add_scenarios(df_med: pd.DataFrame) -> pd.DataFrame:
    df = df_med.copy()
    df["Price_MED"] = df["Price_EUR_MWh"]
    df["Price_LOW"] = LOW_FACTOR * df["Price_MED"]
    df["Price_HIGH"] = HIGH_FACTOR * df["Price_MED"]
    return df

def scenarios_long(df_med: pd.DataFrame) -> pd.DataFrame:
    base = df_med[["date","hour","class","type","regime","Price_EUR_MWh"]].rename(columns={"Price_EUR_MWh":"price"}).copy()

    low = base.copy()
    low["scenario"] = "LOW"
    low["price"] = LOW_FACTOR * low["price"]

    med = base.copy()
    med["scenario"] = "MED"

    high = base.copy()
    high["scenario"] = "HIGH"
    high["price"] = HIGH_FACTOR * high["price"]

    out = pd.concat([low, med, high], ignore_index=True)
    return out[["scenario","date","hour","class","type","regime","price"]].sort_values(["scenario","date","hour"]).reset_index(drop=True)

# ============================================================
# MAE
# ============================================================
def _normalize_eval_df(df_eval: pd.DataFrame) -> pd.DataFrame:
    df = df_eval.copy()

    if "date" not in df.columns:
        raise ValueError("EVAL_CSV necesita columna 'date' (YYYY-MM-DD).")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if "hour" not in df.columns:
        if "Hour" in df.columns:
            hr = pd.to_numeric(df["Hour"], errors="coerce")
            df["hour"] = hr - 1 if hr.max() == 24 else hr
        else:
            raise ValueError("EVAL_CSV necesita 'hour' (0-23) o 'Hour' (1-24).")

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype(int)

    if "Real" not in df.columns:
        raise ValueError("EVAL_CSV necesita columna 'Real'.")

    df["Real"] = pd.to_numeric(df["Real"], errors="coerce")
    return df[["date","hour","Real"]].dropna(subset=["Real"]).copy()

def compute_mae(df_pred: pd.DataFrame, df_eval: pd.DataFrame) -> dict:
    ev = _normalize_eval_df(df_eval)

    merged = ev.merge(df_pred[["date","hour","Price_EUR_MWh"]], on=["date","hour"], how="inner")
    merged = merged.dropna(subset=["Real","Price_EUR_MWh"]).copy()
    merged["abs_err"] = (merged["Price_EUR_MWh"] - merged["Real"]).abs()

    mae_total = float(merged["abs_err"].mean()) if len(merged) else float("nan")

    by_day = merged.groupby("date").agg(
        mae=("abs_err","mean"),
        n=("abs_err","size")
    ).reset_index().sort_values("date")

    by_hour = merged.groupby("hour").agg(
        mae=("abs_err","mean"),
        n=("abs_err","size")
    ).reset_index().sort_values("hour")

    return {"mae_total": mae_total, "merged": merged, "by_day": by_day, "by_hour": by_hour}

# ============================================================
# CLI opcional
# ============================================================
def parse_args_optional():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--hist-dir", default=None)
    p.add_argument("--hist-sep", default=None)
    p.add_argument("--tomorrow-date", default=None)
    p.add_argument("--tomorrow-mean", type=float, default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--out-final", default=None)
    p.add_argument("--out-scen-cols", default=None)
    p.add_argument("--out-scen-long", default=None)
    p.add_argument("--eval-csv", default=None)
    p.add_argument("--eval-sep", default=None)
    p.add_argument("--eval-out-prefix", default=None)
    return p.parse_args()

# ============================================================
# RUN
# ============================================================
def run(hist_dir: str,
        hist_sep: str,
        tomorrow_date: str,
        tomorrow_mean: float,
        days: int,
        out_final: str,
        out_scen_cols: str,
        out_scen_long: str,
        eval_csv: str | None,
        eval_sep: str,
        eval_out_prefix: str):

    df_hist = load_history_csvs(hist_dir, sep=hist_sep)

    week = forecast_week_from_tomorrow_mean(
        df_hist=df_hist,
        tomorrow_date=tomorrow_date,
        tomorrow_mean_eur_mwh=tomorrow_mean,
        days=days,
        low_q=LOW_Q
    )

    tom = week[week["date"] == pd.to_datetime(tomorrow_date)]
    print("OK. Filas forecast:", len(week))
    print("Media mañana (pred):", float(tom["Price_EUR_MWh"].mean()), " target:", float(tomorrow_mean))
    print("k_anchor:", float(week["k_anchor"].iloc[0]), " mu_anchor:", float(week["mu_anchor"].iloc[0]))

    week.to_csv(out_final, index=False)
    print("Guardado:", out_final)

    week_s = add_scenarios(week)
    week_s[[
        "date","hour","class","type","regime",
        "Price_LOW","Price_MED","Price_HIGH",
        "p_mean_pred_eur_mwh","k_anchor","mu_anchor",
        "TomorrowMean_EUR_MWh","LowQuantileUsed","ValleyHours","ValleyMultBase"
    ]].to_csv(out_scen_cols, index=False)
    print("Guardado:", out_scen_cols)

    scenarios_long(week).to_csv(out_scen_long, index=False)
    print("Guardado:", out_scen_long)

    if eval_csv is not None:
        df_eval = pd.read_csv(eval_csv, sep=eval_sep)
        res = compute_mae(week, df_eval)
        print(f"\nMAE total = {res['mae_total']:.4f} €/MWh  (n={len(res['merged'])})")
        res["by_day"].to_csv(f"{eval_out_prefix}by_day.csv", index=False)
        res["by_hour"].to_csv(f"{eval_out_prefix}by_hour.csv", index=False)
        print("Guardados:", f"{eval_out_prefix}by_day.csv,", f"{eval_out_prefix}by_hour.csv")

if __name__ == "__main__":
    args = parse_args_optional()

    hist_dir = args.hist_dir or HIST_DIR
    hist_sep = args.hist_sep or HIST_SEP
    tomorrow_date = args.tomorrow_date or TOMORROW_DATE
    tomorrow_mean = args.tomorrow_mean if args.tomorrow_mean is not None else TOMORROW_MEAN_EUR_MWH
    days = args.days or DAYS

    out_final = args.out_final or OUT_FINAL
    out_scen_cols = args.out_scen_cols or OUT_SCEN_COLS
    out_scen_long = args.out_scen_long or OUT_SCEN_LONG

    eval_csv = args.eval_csv if args.eval_csv is not None else EVAL_CSV
    eval_sep = args.eval_sep or EVAL_SEP
    eval_out_prefix = args.eval_out_prefix or EVAL_OUT_PREFIX

    run(hist_dir, hist_sep, tomorrow_date, tomorrow_mean, days,
        out_final, out_scen_cols, out_scen_long,
        eval_csv, eval_sep, eval_out_prefix)
