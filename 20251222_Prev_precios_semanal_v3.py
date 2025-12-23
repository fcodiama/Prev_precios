#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT ÚNICO (robusto) – Previsión horaria semanal con calibración por MAPE horario
cuando SOLO conoces la media diaria de mañana (y opcionalmente pasado).

✅ Robusto a días incompletos / horas faltantes / duplicadas:
- En backtesting filtra SOLO días completos (24 horas únicas) y salta casos incompletos.
- Agrega duplicados (date,hour) por media.

Entrada:
- Carpeta ./precios con CSVs: 2018.csv ... 2025.csv
- Columnas: Year, Month, Day, Hour (1..24), Value (€/MWh)

Salida (carpeta ./out):
- alpha_profiles_12classes.csv
- class_daily_stats.csv
- calibration_results.csv          (si CALIBRATE=True)
- forecast_week_hourly.csv         (MED)
- forecast_week_hourly_scenarios.csv (si ENABLE_SCENARIOS=True)

Lógica:
- D+1: usa 100% la media diaria ancla (conocida) y desagrega con alpha de clase.
- D+2..: baseline por clase (p50 por defecto) y opcional mezcla con ancla (pesos calibrados).
- Calibración: busca w_D2, w_D3 que minimizan MAPE horario medio D+2..D+7. D+4..D+7=0.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN (EDITA AQUÍ)
# =========================================================

# Media diaria conocida (€/MWh)
TOMORROW_DAILY_MEAN_EUR_MWH = 140.55   # <-- CAMBIA ESTO
DAY_AFTER_DAILY_MEAN_EUR_MWH = None   # <-- opcional (float) o None

HORIZON_DAYS = 7

# Baseline por clase: "p50" (recomendado) o "mu"
BASELINE_STAT = "p50"

# Escenarios LOW/MED/HIGH (p10/p50/p90) para la semana
ENABLE_SCENARIOS = True

# Calibración automática de pesos por MAPE horario
CALIBRATE = True
MAX_TEST_DATES = 1500        # reduce si tarda (800-1500), sube si quieres más precisión
SEED = 123

CANDIDATE_W_D2 = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
CANDIDATE_W_D3 = (0.0, 0.05, 0.1, 0.2, 0.3)

# Si no calibras, pesos por defecto:
ANCHOR_WEIGHTS_DEFAULT = [1.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]

# Carpetas (relativas al script)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "precios"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

# Festivos nacionales fijos (mínimo)
FESTIVOS_FIJOS = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

# Horas internas 0..23; salida en 1..24
INTERNAL_HOUR_0_23 = True

# Seguridad: cap del ancla vs baseline (escala)
S_MIN, S_MAX = 0.6, 1.6

# =========================================================
# CLASES (tipo de día x estación)
# =========================================================

def season_from_month(m: int) -> str:
    if m in (12, 1, 2): return "INV"
    if m in (3, 4, 5):  return "PRI"
    if m in (6, 7, 8):  return "VER"
    return "OTO"

def day_class(ts: pd.Timestamp) -> str:
    m = int(ts.month)
    d = int(ts.day)
    dow = int(ts.weekday())  # lunes=0 ... domingo=6
    holiday = (m, d) in FESTIVOS_FIJOS
    if holiday or dow == 6:
        t = "DOM"
    elif dow == 5:
        t = "SAT"
    else:
        t = "LAB"
    return f"{t}_{season_from_month(m)}"

# =========================================================
# LECTURA Y PREPROCESADO
# =========================================================

def read_one_csv(path: Path) -> pd.DataFrame:
    needed = {"Year", "Month", "Day", "Hour", "Value"}
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if needed.issubset(df.columns):
                return df
        except Exception:
            pass
    raise ValueError(f"No pude leer {path.name} con columnas Year,Month,Day,Hour,Value.")

def build_master_dataframe(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("20*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No encuentro CSVs 20*.csv en {data_dir.resolve()}\n"
            f"Revisa que exista la carpeta y que los ficheros se llamen 2018.csv, 2019.csv, ..."
        )

    df = pd.concat((read_one_csv(f) for f in files), ignore_index=True)
    df = df.rename(columns={
        "Year": "year", "Month": "month", "Day": "day", "Hour": "hour_124", "Value": "price_eur_mwh"
    })

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour_124"] = df["hour_124"].astype(int)

    # Value puede venir como texto con coma decimal
    if df["price_eur_mwh"].dtype == object:
        df["price_eur_mwh"] = df["price_eur_mwh"].astype(str).str.replace(",", ".", regex=False)
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")

    if INTERNAL_HOUR_0_23:
        df["hour"] = df["hour_124"] - 1
        if df["hour"].min() < 0 or df["hour"].max() > 23:
            raise ValueError("Horas fuera de rango al convertir 1..24 a 0..23.")
    else:
        df["hour"] = df["hour_124"]
        if df["hour"].min() < 1 or df["hour"].max() > 24:
            raise ValueError("Horas fuera de rango (esperaba 1..24).")

    # Limpieza mínima
    df = df.dropna(subset=["date", "price_eur_mwh"])
    return df

# =========================================================
# PERFILES ALPHA y STATS DIARIOS POR CLASE
# =========================================================

def compute_alpha_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    alpha_{c,h} = mean_t( p_{t,h}/mean_h(p_{t,h}) ) para t en clase c
    con recorte P5-P95 por (c,h) para evitar extremos.
    """
    tmp = df.copy()
    tmp["class"] = tmp["date"].apply(day_class)

    # Por robustez: agrega duplicados por (date,hour) antes de ratios
    tmp = tmp.groupby(["date", "hour", "class"], as_index=False)["price_eur_mwh"].mean()

    daily_mean = tmp.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day")
    tmp = tmp.merge(daily_mean, on="date", how="left")
    tmp["ratio"] = tmp["price_eur_mwh"] / tmp["p_mean_day"]

    def clip_group(g: pd.DataFrame) -> pd.DataFrame:
        lo = g["ratio"].quantile(0.05)
        hi = g["ratio"].quantile(0.95)
        g["ratio"] = g["ratio"].clip(lo, hi)
        return g

    tmp = tmp.groupby(["class", "hour"], group_keys=False).apply(clip_group)

    alpha = (
        tmp.groupby(["class", "hour"])["ratio"]
           .mean()
           .reset_index()
           .rename(columns={"ratio": "alpha"})
    )
    return alpha

def compute_class_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["class"] = tmp["date"].apply(day_class)

    # Agrega duplicados (date,hour)
    tmp = tmp.groupby(["date", "hour", "class"], as_index=False)["price_eur_mwh"].mean()

    daily = tmp.groupby(["date", "class"])["price_eur_mwh"].mean().reset_index(name="p_mean_day")
    stats = (
        daily.groupby("class")["p_mean_day"]
             .agg(mu="mean",
                  p10=lambda x: x.quantile(0.10),
                  p50=lambda x: x.quantile(0.50),
                  p90=lambda x: x.quantile(0.90),
                  n_days="count")
             .reset_index()
    )
    return stats

def alpha_for_class_or_global(alpha: pd.DataFrame, c: str) -> pd.DataFrame:
    a = alpha[alpha["class"] == c].sort_values("hour")
    if len(a) == 24:
        return a
    return alpha.groupby("hour", as_index=False)["alpha"].mean().sort_values("hour")

def get_stat(class_stats: pd.DataFrame, c: str, stat: str) -> float:
    row = class_stats.loc[class_stats["class"] == c, stat]
    if row.empty:
        raise ValueError(f"No hay {stat} para la clase {c}.")
    return float(row.iloc[0])

def blended_daily_mean(anchor_mean: float, baseline_mean: float, w: float) -> float:
    """mean = w*anchor + (1-w)*baseline, con cap de escala anchor/baseline."""
    if w <= 0.0:
        return float(baseline_mean)
    if baseline_mean <= 0:
        return float(anchor_mean)
    s = float(np.clip(anchor_mean / baseline_mean, S_MIN, S_MAX))
    anchor_capped = baseline_mean * s
    return float(w * anchor_capped + (1.0 - w) * baseline_mean)

# =========================================================
# BACKTEST + CALIBRACIÓN (MAPE horario)
# =========================================================

def mape(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denom))

def build_daily_and_hourly_truth(df: pd.DataFrame):
    """
    Devuelve daily y hourly_true SOLO para días completos (24 horas únicas).
    También agrega duplicados (date,hour) por media.
    """
    tmp = df.groupby(["date", "hour"], as_index=False)["price_eur_mwh"].mean()

    counts = tmp.groupby("date")["hour"].nunique()
    full_days = counts[counts == 24].index
    tmp = tmp[tmp["date"].isin(full_days)].copy()

    daily = tmp.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day").to_frame()
    hourly_true = tmp.set_index(["date", "hour"])["price_eur_mwh"].sort_index()
    return daily, hourly_true

def forecast_day_from_anchor(date_target: pd.Timestamp,
                             anchor_mean: float,
                             alpha: pd.DataFrame,
                             class_stats: pd.DataFrame,
                             baseline_stat: str,
                             w_anchor: float) -> np.ndarray:
    c = day_class(date_target)
    baseline = get_stat(class_stats, c, baseline_stat)
    daily_mean = blended_daily_mean(anchor_mean, baseline, w_anchor)
    a = alpha_for_class_or_global(alpha, c).sort_values("hour")
    return daily_mean * a["alpha"].to_numpy()

def calibrate_anchor_weights(df: pd.DataFrame,
                             alpha: pd.DataFrame,
                             class_stats: pd.DataFrame,
                             baseline_stat: str,
                             candidate_w_d2,
                             candidate_w_d3,
                             horizon_days: int,
                             max_test_dates: int,
                             seed: int) -> pd.DataFrame:
    """
    Calibra w_D2 y w_D3. D+1 fijo 1.0 (ancla), D+4..D+7 fijo 0.
    Score: MAPE horario medio D+2..D+7.
    """
    daily, hourly_true = build_daily_and_hourly_truth(df)
    dates = daily.index.sort_values()

    # d0 es "hoy"; necesitamos que existan d0+1..d0+H en daily (días completos)
    valid = []
    daily_index = set(daily.index)
    for d0 in dates:
        ok = True
        for k in range(1, horizon_days + 1):
            if (d0 + pd.Timedelta(days=k)) not in daily_index:
                ok = False
                break
        if ok:
            valid.append(d0)
    valid = pd.Index(valid).sort_values()

    if len(valid) == 0:
        raise RuntimeError("No hay fechas válidas con horizonte completo (días completos). Revisa datos.")

    rng = np.random.default_rng(seed)
    if len(valid) > max_test_dates:
        valid = pd.Index(rng.choice(valid, size=max_test_dates, replace=False)).sort_values()

    results = []
    for w2 in candidate_w_d2:
        for w3 in candidate_w_d3:
            weights = [1.0, float(w2), float(w3)] + [0.0] * (horizon_days - 3)

            mape_by_k = {k: [] for k in range(1, horizon_days + 1)}
            target_mapes = []

            for d0 in valid:
                d1 = d0 + pd.Timedelta(days=1)
                # ancla: media real de D+1 (d1) (días completos garantizados)
                anchor_mean = float(daily.loc[d1, "p_mean_day"])

                for k in range(1, horizon_days + 1):
                    dt = d0 + pd.Timedelta(days=k)
                    w = 1.0 if k == 1 else float(weights[k-1])

                    pred = forecast_day_from_anchor(dt, anchor_mean, alpha, class_stats, baseline_stat, w)

                    # verdad 24h robusta
                    idx = pd.MultiIndex.from_product([[dt], range(24)], names=["date", "hour"])
                    true = hourly_true.reindex(idx).to_numpy(dtype=float)
                    if np.isnan(true).any():
                        # no debería ocurrir (full days), pero por seguridad:
                        continue

                    mk = mape(true, pred)
                    mape_by_k[k].append(mk)
                    if k >= 2:
                        target_mapes.append(mk)

            score = float(np.mean(target_mapes)) if target_mapes else np.inf
            row = {"w_D2": float(w2), "w_D3": float(w3), "MAPE_D2_D7": score}
            for k in range(1, horizon_days + 1):
                row[f"MAPE_D{k}"] = float(np.mean(mape_by_k[k])) if mape_by_k[k] else np.nan
            results.append(row)

    return pd.DataFrame(results).sort_values("MAPE_D2_D7").reset_index(drop=True)

# =========================================================
# FORECAST SEMANAL (MED y escenarios)
# =========================================================

def forecast_week(alpha: pd.DataFrame,
                  class_stats: pd.DataFrame,
                  tomorrow_date: pd.Timestamp,
                  mean_tom: float,
                  mean_day_after: float | None,
                  days: int,
                  baseline_stat: str,
                  anchor_weights: list[float],
                  scenario_label: str,
                  scenario_stat: str) -> pd.DataFrame:
    if len(anchor_weights) < days:
        anchor_weights = anchor_weights + [0.0] * (days - len(anchor_weights))

    rows = []
    for i in range(days):
        d = tomorrow_date + pd.Timedelta(days=i)
        c = day_class(d)

        # Media diaria objetivo
        if i == 0:
            daily_mean = float(mean_tom)
            src = "ANCHOR_TOMORROW_MEAN"
        elif i == 1 and mean_day_after is not None:
            daily_mean = float(mean_day_after)
            src = "ANCHOR_DAY_AFTER_MEAN"
        else:
            # baseline por clase según escenario
            base_stat = baseline_stat if scenario_label == "MED" else scenario_stat
            baseline = get_stat(class_stats, c, base_stat)
            w = float(anchor_weights[i])
            daily_mean = blended_daily_mean(anchor_mean=float(mean_tom), baseline_mean=float(baseline), w=w)
            src = f"BLEND_w={w:.2f}_baseline={base_stat}"

        a = alpha_for_class_or_global(alpha, c).sort_values("hour")
        prices = daily_mean * a["alpha"].to_numpy()

        for hour, price in zip(a["hour"].to_numpy(), prices):
            out_hour = int(hour) + 1 if INTERNAL_HOUR_0_23 else int(hour)
            rows.append({
                "Scenario": scenario_label,
                "Date": d.date().isoformat(),
                "Hour": out_hour,  # 1..24
                "DailyMeanUsed_EUR_MWh": daily_mean,
                "Price_EUR_MWh": float(price),
                "Class": c,
                "Source": src
            })
    return pd.DataFrame(rows)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Buscando CSVs en:", DATA_DIR.resolve())
    df = build_master_dataframe(DATA_DIR)

    # Diagnóstico mínimo de integridad
    tmp = df.groupby(["date", "hour"], as_index=False)["price_eur_mwh"].mean()
    counts = tmp.groupby("date")["hour"].nunique()
    print("Días totales (en histórico):", counts.size)
    print("Días completos (24h únicas):", int((counts == 24).sum()))
    print("Días incompletos:", int((counts != 24).sum()))

    alpha = compute_alpha_profiles(df)
    class_stats = compute_class_daily_stats(df)

    alpha_path = OUT_DIR / "alpha_profiles_12classes.csv"
    stats_path = OUT_DIR / "class_daily_stats.csv"
    alpha.to_csv(alpha_path, index=False)
    class_stats.to_csv(stats_path, index=False)
    print("OK ->", alpha_path)
    print("OK ->", stats_path)

    # Calibración
    if CALIBRATE:
        print("\nCalibrando pesos (MAPE horario)...")
        calib = calibrate_anchor_weights(
            df=df,
            alpha=alpha,
            class_stats=class_stats,
            baseline_stat=BASELINE_STAT,
            candidate_w_d2=CANDIDATE_W_D2,
            candidate_w_d3=CANDIDATE_W_D3,
            horizon_days=HORIZON_DAYS,
            max_test_dates=MAX_TEST_DATES,
            seed=SEED
        )
        calib_path = OUT_DIR / "calibration_results.csv"
        calib.to_csv(calib_path, index=False)
        best = calib.iloc[0]
        anchor_weights = [1.0, float(best["w_D2"]), float(best["w_D3"])] + [0.0] * (HORIZON_DAYS - 3)

        print("OK ->", calib_path)
        print("Mejor (MAPE D+2..D+7):", best.to_dict())
        print("ANCHOR_WEIGHTS usados:", anchor_weights)
    else:
        anchor_weights = ANCHOR_WEIGHTS_DEFAULT[:HORIZON_DAYS]
        print("\nSin calibración. ANCHOR_WEIGHTS:", anchor_weights)

    # Forecast
    tomorrow = pd.Timestamp((datetime.now().date() + timedelta(days=1)).isoformat())
    print("\n--- FORECAST ---")
    print("Inicio (mañana):", tomorrow.date().isoformat(), "| Clase:", day_class(tomorrow))
    print("Media diaria mañana (€/MWh):", TOMORROW_DAILY_MEAN_EUR_MWH)
    print("Media diaria pasado (€/MWh):", DAY_AFTER_DAILY_MEAN_EUR_MWH)
    print("Baseline stat:", BASELINE_STAT)
    print("Scenarios:", ENABLE_SCENARIOS)

    # MED
    fc_med = forecast_week(
        alpha=alpha,
        class_stats=class_stats,
        tomorrow_date=tomorrow,
        mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
        mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
        days=HORIZON_DAYS,
        baseline_stat=BASELINE_STAT,
        anchor_weights=anchor_weights,
        scenario_label="MED",
        scenario_stat="p50"
    )
    out_med = OUT_DIR / "forecast_week_hourly.csv"
    fc_med.to_csv(out_med, index=False)
    print("OK ->", out_med)

    # Escenarios
    if ENABLE_SCENARIOS:
        scen = [("LOW", "p10"), ("MED", "p50"), ("HIGH", "p90")]
        fc_all = []
        for lab, stat in scen:
            fc_all.append(forecast_week(
                alpha=alpha,
                class_stats=class_stats,
                tomorrow_date=tomorrow,
                mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
                mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
                days=HORIZON_DAYS,
                baseline_stat=BASELINE_STAT,
                anchor_weights=anchor_weights,
                scenario_label=lab,
                scenario_stat=stat
            ))
        fc_s = pd.concat(fc_all, ignore_index=True)
        out_s = OUT_DIR / "forecast_week_hourly_scenarios.csv"
        fc_s.to_csv(out_s, index=False)
        print("OK ->", out_s)

    print("\nListo.")
    print("Si la calibración tarda: baja MAX_TEST_DATES (p.ej. 800).")
