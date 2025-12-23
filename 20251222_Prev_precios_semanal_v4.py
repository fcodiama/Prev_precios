#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast semanal horario cuando SOLO conoces la media diaria de mañana (y opcionalmente pasado),
con calibración por MAPE horario (rápida + con progreso) y robustez a días incompletos.

Entrada:
- ./precios/2018.csv ... 2025.csv
  Columnas: Year, Month, Day, Hour(1..24), Value(€/MWh)

Salida:
- ./out/alpha_profiles_12classes.csv
- ./out/class_daily_stats.csv
- ./out/calibration_results.csv (si CALIBRATE=True)
- ./out/forecast_week_hourly.csv
- ./out/forecast_week_hourly_scenarios.csv (si ENABLE_SCENARIOS=True)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# =========================
# CONFIG (EDITA AQUÍ)
# =========================
TOMORROW_DAILY_MEAN_EUR_MWH = 140.55
DAY_AFTER_DAILY_MEAN_EUR_MWH = None

HORIZON_DAYS = 7
BASELINE_STAT = "p50"               # "p50" recomendado
ENABLE_SCENARIOS = True

CALIBRATE = True
MAX_TEST_DATES = 800                # 👈 BAJA ESTO si tarda (200-800). Sube si quieres.
SEED = 123

CANDIDATE_W_D2 = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
CANDIDATE_W_D3 = (0.0, 0.05, 0.1, 0.2, 0.3)
ANCHOR_WEIGHTS_DEFAULT = [1.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "precios"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

FESTIVOS_FIJOS = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

INTERNAL_HOUR_0_23 = True
S_MIN, S_MAX = 0.6, 1.6

# =========================
# CLASES
# =========================
def season_from_month(m: int) -> str:
    if m in (12, 1, 2): return "INV"
    if m in (3, 4, 5):  return "PRI"
    if m in (6, 7, 8):  return "VER"
    return "OTO"

def day_class(ts: pd.Timestamp) -> str:
    m = int(ts.month)
    d = int(ts.day)
    dow = int(ts.weekday())
    holiday = (m, d) in FESTIVOS_FIJOS
    if holiday or dow == 6:
        t = "DOM"
    elif dow == 5:
        t = "SAT"
    else:
        t = "LAB"
    return f"{t}_{season_from_month(m)}"

# =========================
# LECTURA
# =========================
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
        raise FileNotFoundError(f"No encuentro CSVs 20*.csv en {data_dir.resolve()}")
    df = pd.concat((read_one_csv(f) for f in files), ignore_index=True)

    df = df.rename(columns={"Year":"year","Month":"month","Day":"day","Hour":"hour_124","Value":"price_eur_mwh"})
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour_124"] = df["hour_124"].astype(int)

    if df["price_eur_mwh"].dtype == object:
        df["price_eur_mwh"] = df["price_eur_mwh"].astype(str).str.replace(",", ".", regex=False)
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")

    if INTERNAL_HOUR_0_23:
        df["hour"] = df["hour_124"] - 1
    else:
        df["hour"] = df["hour_124"]

    df = df.dropna(subset=["date","price_eur_mwh"])
    return df

# =========================
# ALPHA + STATS
# =========================
def compute_alpha_profiles(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["class"] = tmp["date"].apply(day_class)
    tmp = tmp.groupby(["date","hour","class"], as_index=False)["price_eur_mwh"].mean()

    daily_mean = tmp.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day")
    tmp = tmp.merge(daily_mean, on="date", how="left")
    tmp["ratio"] = tmp["price_eur_mwh"] / tmp["p_mean_day"]

    def clip_group(g):
        lo = g["ratio"].quantile(0.05)
        hi = g["ratio"].quantile(0.95)
        g["ratio"] = g["ratio"].clip(lo, hi)
        return g
    tmp = tmp.groupby(["class","hour"], group_keys=False).apply(clip_group)

    alpha = (tmp.groupby(["class","hour"])["ratio"].mean()
             .reset_index().rename(columns={"ratio":"alpha"}))
    return alpha

def compute_class_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["class"] = tmp["date"].apply(day_class)
    tmp = tmp.groupby(["date","hour","class"], as_index=False)["price_eur_mwh"].mean()
    daily = tmp.groupby(["date","class"])["price_eur_mwh"].mean().reset_index(name="p_mean_day")
    stats = (daily.groupby("class")["p_mean_day"]
             .agg(mu="mean",
                  p10=lambda x: x.quantile(0.10),
                  p50=lambda x: x.quantile(0.50),
                  p90=lambda x: x.quantile(0.90),
                  n_days="count")
             .reset_index())
    return stats

def alpha_for_class_or_global(alpha: pd.DataFrame, c: str) -> np.ndarray:
    a = alpha[alpha["class"] == c].sort_values("hour")
    if len(a) == 24:
        return a["alpha"].to_numpy()
    return alpha.groupby("hour", as_index=False)["alpha"].mean().sort_values("hour")["alpha"].to_numpy()

def get_stat(stats: pd.DataFrame, c: str, stat: str) -> float:
    v = stats.loc[stats["class"] == c, stat]
    return float(v.iloc[0])

def blended_daily_mean(anchor_mean: float, baseline_mean: float, w: float) -> float:
    if w <= 0.0: return float(baseline_mean)
    if baseline_mean <= 0: return float(anchor_mean)
    s = float(np.clip(anchor_mean / baseline_mean, S_MIN, S_MAX))
    anchor_capped = baseline_mean * s
    return float(w * anchor_capped + (1.0 - w) * baseline_mean)

# =========================
# CALIBRACIÓN (RÁPIDA + PROGRESO)
# =========================
def mape_vec(true_mat: np.ndarray, pred_mat: np.ndarray, eps=1e-6) -> float:
    # true_mat/pred_mat: (N,24)
    denom = np.maximum(np.abs(true_mat), eps)
    return float(100.0 * np.mean(np.abs((true_mat - pred_mat) / denom)))

def build_truth_matrices(df: pd.DataFrame, horizon_days: int):
    """
    Construye matrices para backtest:
    - Solo días completos (24 horas únicas)
    - Para cada d0 válido, construye verdad de k=1..H en matrices
    Devuelve:
      d0_list, anchor_mean_list, true_by_k[k] (N,24)
    """
    tmp = df.groupby(["date","hour"], as_index=False)["price_eur_mwh"].mean()
    counts = tmp.groupby("date")["hour"].nunique()
    full_days = counts[counts == 24].index
    tmp = tmp[tmp["date"].isin(full_days)].copy()

    daily = tmp.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day")
    hourly = tmp.pivot(index="date", columns="hour", values="price_eur_mwh").sort_index()
    # hourly: index=date, columns=0..23

    dates = hourly.index
    daily_index = set(dates)

    valid = []
    for d0 in dates:
        ok = True
        for k in range(1, horizon_days + 1):
            if (d0 + pd.Timedelta(days=k)) not in daily_index:
                ok = False; break
        if ok:
            valid.append(d0)
    valid = pd.Index(valid).sort_values()
    return daily, hourly, valid

def calibrate_anchor_weights_fast(df: pd.DataFrame, alpha: pd.DataFrame, stats: pd.DataFrame,
                                  baseline_stat: str, candidate_w_d2, candidate_w_d3,
                                  horizon_days: int, max_test_dates: int, seed: int) -> pd.DataFrame:
    daily, hourly, valid = build_truth_matrices(df, horizon_days)

    rng = np.random.default_rng(seed)
    if len(valid) > max_test_dates:
        valid = pd.Index(rng.choice(valid, size=max_test_dates, replace=False)).sort_values()

    # Precompute: classes and alpha vectors for all needed dates
    # We'll need dt = d0 + k
    needed_dates = set()
    for d0 in valid:
        for k in range(1, horizon_days+1):
            needed_dates.add(d0 + pd.Timedelta(days=k))
    needed_dates = sorted(needed_dates)

    class_map = {d: day_class(d) for d in needed_dates}
    alpha_map = {c: alpha_for_class_or_global(alpha, c) for c in set(class_map.values())}

    # Precompute baseline for each date for p50/mu
    baseline_map = {d: get_stat(stats, class_map[d], baseline_stat) for d in needed_dates}

    # True matrices by horizon k: (N,24)
    true_by_k = {}
    for k in range(1, horizon_days+1):
        dt_list = [d0 + pd.Timedelta(days=k) for d0 in valid]
        true_by_k[k] = hourly.loc[dt_list].to_numpy(dtype=float)

    # Anchors: for each d0, anchor is daily mean of d1=d0+1
    anchors = np.array([float(daily.loc[d0 + pd.Timedelta(days=1)]) for d0 in valid], dtype=float)

    # For each k>=2 we also need baseline per dt and alpha vector per dt
    baseline_by_k = {}
    alpha_by_k = {}
    for k in range(1, horizon_days+1):
        dt_list = [d0 + pd.Timedelta(days=k) for d0 in valid]
        baseline_by_k[k] = np.array([baseline_map[dt] for dt in dt_list], dtype=float)  # (N,)
        alpha_by_k[k] = np.stack([alpha_map[class_map[dt]] for dt in dt_list], axis=0)  # (N,24)

    results = []
    combos = [(w2, w3) for w2 in candidate_w_d2 for w3 in candidate_w_d3]
    total = len(combos)
    print(f"Calibración: probando {total} combinaciones (N={len(valid)} fechas, H={horizon_days}).")

    for idx, (w2, w3) in enumerate(combos, start=1):
        weights = {1: 1.0, 2: float(w2), 3: float(w3)}
        # k>=4 -> 0
        target_mapes = []
        mape_k = {}

        for k in range(1, horizon_days+1):
            w = weights.get(k, 0.0)

            # daily_mean_pred for each sample (N,)
            if k == 1:
                # D+1: usamos la media ancla (anchors) directamente y desagregamos con alpha del dt
                daily_mean_pred = anchors.copy()
            else:
                baseline = baseline_by_k[k]
                # mezcla con cap: anchor_capped = baseline * clip(anchor/baseline)
                s = np.clip(anchors / np.maximum(baseline, 1e-9), S_MIN, S_MAX)
                anchor_capped = baseline * s
                daily_mean_pred = w * anchor_capped + (1.0 - w) * baseline

            pred = daily_mean_pred[:, None] * alpha_by_k[k]  # (N,24)
            mk = mape_vec(true_by_k[k], pred)
            mape_k[k] = mk
            if k >= 2:
                target_mapes.append(mk)

        score = float(np.mean(target_mapes))
        row = {"w_D2": float(w2), "w_D3": float(w3), "MAPE_D2_D7": score}
        for k in range(1, horizon_days+1):
            row[f"MAPE_D{k}"] = mape_k[k]
        results.append(row)

        if idx % max(1, total//10) == 0 or idx == 1 or idx == total:
            print(f"  Progreso {idx}/{total} | mejor MAPE_D2_D7 hasta ahora: {min(r['MAPE_D2_D7'] for r in results):.3f}")

    return pd.DataFrame(results).sort_values("MAPE_D2_D7").reset_index(drop=True)

# =========================
# FORECAST SEMANAL
# =========================
def forecast_week(alpha: pd.DataFrame, stats: pd.DataFrame,
                  tomorrow_date: pd.Timestamp, mean_tom: float, mean_day_after: float | None,
                  days: int, baseline_stat: str, anchor_weights: list[float],
                  scenario_label: str, scenario_stat: str) -> pd.DataFrame:

    if len(anchor_weights) < days:
        anchor_weights = anchor_weights + [0.0] * (days - len(anchor_weights))

    rows = []
    for i in range(days):
        d = tomorrow_date + pd.Timedelta(days=i)
        c = day_class(d)
        a = alpha_for_class_or_global(alpha, c)

        if i == 0:
            daily_mean = float(mean_tom)
            src = "ANCHOR_TOMORROW_MEAN"
        elif i == 1 and mean_day_after is not None:
            daily_mean = float(mean_day_after)
            src = "ANCHOR_DAY_AFTER_MEAN"
        else:
            base_stat = baseline_stat if scenario_label == "MED" else scenario_stat
            baseline = get_stat(stats, c, base_stat)
            w = float(anchor_weights[i])
            daily_mean = blended_daily_mean(anchor_mean=float(mean_tom), baseline_mean=float(baseline), w=w)
            src = f"BLEND_w={w:.2f}_baseline={base_stat}"

        prices = daily_mean * a
        for h0, p in enumerate(prices):
            hour_out = h0 + 1 if INTERNAL_HOUR_0_23 else h0
            rows.append({
                "Scenario": scenario_label,
                "Date": d.date().isoformat(),
                "Hour": int(hour_out),
                "DailyMeanUsed_EUR_MWh": float(daily_mean),
                "Price_EUR_MWh": float(p),
                "Class": c,
                "Source": src
            })
    return pd.DataFrame(rows)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Buscando CSVs en:", DATA_DIR.resolve())
    df = build_master_dataframe(DATA_DIR)

    # diagnóstico rápido
    tmp = df.groupby(["date","hour"], as_index=False)["price_eur_mwh"].mean()
    counts = tmp.groupby("date")["hour"].nunique()
    print("Días totales:", counts.size,
          "| completos(24h):", int((counts==24).sum()),
          "| incompletos:", int((counts!=24).sum()))

    alpha = compute_alpha_profiles(df)
    stats = compute_class_daily_stats(df)

    alpha.to_csv(OUT_DIR / "alpha_profiles_12classes.csv", index=False)
    stats.to_csv(OUT_DIR / "class_daily_stats.csv", index=False)
    print("OK -> alpha_profiles_12classes.csv")
    print("OK -> class_daily_stats.csv")

    # calibración
    if CALIBRATE:
        print("\nCalibrando pesos (MAPE horario)...")
        calib = calibrate_anchor_weights_fast(
            df=df, alpha=alpha, stats=stats,
            baseline_stat=BASELINE_STAT,
            candidate_w_d2=CANDIDATE_W_D2,
            candidate_w_d3=CANDIDATE_W_D3,
            horizon_days=HORIZON_DAYS,
            max_test_dates=MAX_TEST_DATES,
            seed=SEED
        )
        calib.to_csv(OUT_DIR / "calibration_results.csv", index=False)
        best = calib.iloc[0]
        anchor_weights = [1.0, float(best["w_D2"]), float(best["w_D3"])] + [0.0]*(HORIZON_DAYS-3)
        print("OK -> calibration_results.csv")
        print("Mejor:", best.to_dict())
        print("ANCHOR_WEIGHTS:", anchor_weights)
    else:
        anchor_weights = ANCHOR_WEIGHTS_DEFAULT[:HORIZON_DAYS]
        print("Sin calibración. ANCHOR_WEIGHTS:", anchor_weights)

    tomorrow = pd.Timestamp((datetime.now().date() + timedelta(days=1)).isoformat())
    print("\n--- FORECAST ---")
    print("Inicio:", tomorrow.date().isoformat(), "| Clase:", day_class(tomorrow))
    print("Media diaria mañana:", TOMORROW_DAILY_MEAN_EUR_MWH, "€/MWh")

    # MED
    fc_med = forecast_week(
        alpha=alpha, stats=stats,
        tomorrow_date=tomorrow,
        mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
        mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
        days=HORIZON_DAYS,
        baseline_stat=BASELINE_STAT,
        anchor_weights=anchor_weights,
        scenario_label="MED",
        scenario_stat="p50"
    )
    fc_med.to_csv(OUT_DIR / "forecast_week_hourly.csv", index=False)
    print("OK -> forecast_week_hourly.csv")

    if ENABLE_SCENARIOS:
        all_fc = []
        for lab, stat in [("LOW","p10"), ("MED","p50"), ("HIGH","p90")]:
            all_fc.append(forecast_week(
                alpha=alpha, stats=stats,
                tomorrow_date=tomorrow,
                mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
                mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
                days=HORIZON_DAYS,
                baseline_stat=BASELINE_STAT,
                anchor_weights=anchor_weights,
                scenario_label=lab,
                scenario_stat=stat
            ))
        pd.concat(all_fc, ignore_index=True).to_csv(OUT_DIR / "forecast_week_hourly_scenarios.csv", index=False)
        print("OK -> forecast_week_hourly_scenarios.csv")

    print("\nListo. Si sigue tardando, baja MAX_TEST_DATES a 200-400.")
