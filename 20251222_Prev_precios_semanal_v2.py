#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast semanal horario cuando SOLO conoces la media diaria de mañana (y opcionalmente pasado),
con mejora para D+2..D+7:

- D+1 (mañana): usa 100% la media diaria ANCLA (conocida) y desagrega a horas con alpha_clase.
- D+2: mezcla (blend) entre ANCLA y baseline de clase (mu o p50) para evitar arrastre excesivo.
- D+3+: vuelve rápidamente al baseline (mu/p50) para no degradar tanto.

Además genera 3 escenarios (LOW/MED/HIGH) para D+2.. (opcional):
- LOW  = p10 de la clase
- MED  = p50 de la clase
- HIGH = p90 de la clase
y aplica mezcla con el ancla SOLO en D+2 (por defecto).

Entrada:
- CSVs en ./precios: 2018.csv ... 2025.csv
  Columnas: Year,Month,Day,Hour (1..24),Value (€/MWh)

Salida:
- out/alpha_profiles_12classes.csv
- out/class_daily_stats.csv
- out/forecast_week_hourly.csv  (escenario MED por defecto)
- out/forecast_week_hourly_scenarios.csv (si ENABLE_SCENARIOS=True)
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
DAY_AFTER_DAILY_MEAN_EUR_MWH = None   # <-- opcional

HORIZON_DAYS = 7

# Mezcla con el ancla (peso del ancla por día i=0..HORIZON_DAYS-1)
# Recomendado para tu caso (volver rápido a baseline):
#   i=0 (mañana): 1.00
#   i=1 (pasado): 0.50
#   i>=2:         0.00
ANCHOR_WEIGHTS = [1.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00]

# Baseline diario a usar por clase: "mu" o "p50"
BASELINE_STAT = "p50"   # <- p50 suele ser más robusto que mu

# Escenarios para D+2..: p10/p50/p90 por clase
ENABLE_SCENARIOS = True

# Carpetas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "precios"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

# Festivos nacionales fijos (mínimo)
FESTIVOS_FIJOS = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

# Horas internas 0..23, salida 1..24
INTERNAL_HOUR_0_23 = True

# Seguridad: límites de escala del ancla vs baseline (para evitar días raros)
S_MIN, S_MAX = 0.6, 1.6

# =========================================================
# HELPERS
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

    df = df.dropna(subset=["date", "price_eur_mwh"])
    return df

def compute_alpha_profiles(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["class"] = tmp["date"].apply(day_class)

    daily_mean = tmp.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day")
    tmp = tmp.merge(daily_mean, on="date", how="left")
    tmp["ratio"] = tmp["price_eur_mwh"] / tmp["p_mean_day"]

    # recorte P5-P95 por (class,hour) para evitar extremos
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
    """
    Mezcla controlada:
      mean = w*anchor + (1-w)*baseline
    pero limitando la escala anchor/baseline.
    """
    if baseline_mean <= 0:
        return anchor_mean
    s = anchor_mean / baseline_mean
    s = float(np.clip(s, S_MIN, S_MAX))
    anchor_capped = baseline_mean * s
    return float(w * anchor_capped + (1.0 - w) * baseline_mean)

def forecast_week(alpha: pd.DataFrame,
                  class_stats: pd.DataFrame,
                  tomorrow_date: pd.Timestamp,
                  mean_tom: float,
                  mean_day_after: float | None,
                  days: int = HORIZON_DAYS,
                  baseline_stat: str = BASELINE_STAT,
                  anchor_weights: list[float] = ANCHOR_WEIGHTS) -> pd.DataFrame:

    if len(anchor_weights) < days:
        anchor_weights = anchor_weights + [0.0] * (days - len(anchor_weights))

    rows = []
    for i in range(days):
        d = tomorrow_date + pd.Timedelta(days=i)
        c = day_class(d)
        w = float(anchor_weights[i])

        # baseline (mu o p50) por clase
        baseline = get_stat(class_stats, c, baseline_stat)

        # ancla disponible para i=0 y opcional i=1
        if i == 0:
            daily_mean = float(mean_tom)  # 100% conocido, no mezclar
            source = "ANCHOR_TOMORROW_MEAN"
        elif i == 1 and mean_day_after is not None:
            daily_mean = float(mean_day_after)
            source = "ANCHOR_DAY_AFTER_MEAN"
        else:
            # mezcla con el ancla de mañana (único ancla disponible) si w>0
            daily_mean = blended_daily_mean(anchor_mean=float(mean_tom), baseline_mean=baseline, w=w)
            source = f"BLEND_w={w:.2f}_baseline={baseline_stat}"

        # desagregación horaria con alpha de clase
        a = alpha_for_class_or_global(alpha, c)
        prices = daily_mean * a["alpha"].to_numpy()

        for hour, price in zip(a["hour"].to_numpy(), prices):
            out_hour = int(hour) + 1 if INTERNAL_HOUR_0_23 else int(hour)
            rows.append({
                "Scenario": "MED",
                "Date": d.date().isoformat(),
                "Hour": out_hour,
                "DailyMeanUsed_EUR_MWh": daily_mean,
                "Price_EUR_MWh": float(price),
                "Class": c,
                "Source": source
            })

    return pd.DataFrame(rows)

def forecast_week_scenarios(alpha: pd.DataFrame,
                            class_stats: pd.DataFrame,
                            tomorrow_date: pd.Timestamp,
                            mean_tom: float,
                            mean_day_after: float | None,
                            days: int = HORIZON_DAYS,
                            anchor_weights: list[float] = ANCHOR_WEIGHTS) -> pd.DataFrame:
    """
    Escenarios LOW/MED/HIGH:
    - i=0: se respeta el ancla (solo hay una media conocida; mismo para los 3 escenarios)
    - i=1: si hay ancla de pasado, se respeta (igual para los 3)
    - i>=2: baseline por clase usa p10/p50/p90 (y mezcla con ancla según w)
    """
    if len(anchor_weights) < days:
        anchor_weights = anchor_weights + [0.0] * (days - len(anchor_weights))

    scenario_stat = {"LOW": "p10", "MED": "p50", "HIGH": "p90"}

    rows = []
    for scen, stat in scenario_stat.items():
        for i in range(days):
            d = tomorrow_date + pd.Timedelta(days=i)
            c = day_class(d)
            w = float(anchor_weights[i])

            baseline = get_stat(class_stats, c, stat)

            if i == 0:
                daily_mean = float(mean_tom)
                source = "ANCHOR_TOMORROW_MEAN"
            elif i == 1 and mean_day_after is not None:
                daily_mean = float(mean_day_after)
                source = "ANCHOR_DAY_AFTER_MEAN"
            else:
                daily_mean = blended_daily_mean(anchor_mean=float(mean_tom), baseline_mean=baseline, w=w)
                source = f"BLEND_w={w:.2f}_baseline={stat}"

            a = alpha_for_class_or_global(alpha, c)
            prices = daily_mean * a["alpha"].to_numpy()

            for hour, price in zip(a["hour"].to_numpy(), prices):
                out_hour = int(hour) + 1 if INTERNAL_HOUR_0_23 else int(hour)
                rows.append({
                    "Scenario": scen,
                    "Date": d.date().isoformat(),
                    "Hour": out_hour,
                    "DailyMeanUsed_EUR_MWh": daily_mean,
                    "Price_EUR_MWh": float(price),
                    "Class": c,
                    "Source": source
                })

    return pd.DataFrame(rows)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Leyendo histórico desde:", DATA_DIR.resolve())
    df = build_master_dataframe(DATA_DIR)

    alpha = compute_alpha_profiles(df)
    class_stats = compute_class_daily_stats(df)

    alpha_path = OUT_DIR / "alpha_profiles_12classes.csv"
    stats_path = OUT_DIR / "class_daily_stats.csv"
    alpha.to_csv(alpha_path, index=False)
    class_stats.to_csv(stats_path, index=False)

    print("OK ->", alpha_path)
    print("OK ->", stats_path)

    tomorrow = pd.Timestamp((datetime.now().date() + timedelta(days=1)).isoformat())
    print("\n--- CONFIG ---")
    print("Mañana:", tomorrow.date().isoformat(), "| Clase:", day_class(tomorrow))
    print("Media diaria mañana (€/MWh):", TOMORROW_DAILY_MEAN_EUR_MWH)
    print("Media diaria pasado (€/MWh):", DAY_AFTER_DAILY_MEAN_EUR_MWH)
    print("Anchor weights:", ANCHOR_WEIGHTS[:HORIZON_DAYS])
    print("Baseline stat:", BASELINE_STAT)
    print("Scenarios:", ENABLE_SCENARIOS)

    # Forecast MED (una sola serie)
    fc_med = forecast_week(
        alpha=alpha,
        class_stats=class_stats,
        tomorrow_date=tomorrow,
        mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
        mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
        days=HORIZON_DAYS,
        baseline_stat=BASELINE_STAT,
        anchor_weights=ANCHOR_WEIGHTS
    )

    out_med = OUT_DIR / "forecast_week_hourly.csv"
    fc_med.to_csv(out_med, index=False)
    print("OK ->", out_med)

    # Escenarios
    if ENABLE_SCENARIOS:
        fc_s = forecast_week_scenarios(
            alpha=alpha,
            class_stats=class_stats,
            tomorrow_date=tomorrow,
            mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
            mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
            days=HORIZON_DAYS,
            anchor_weights=ANCHOR_WEIGHTS
        )
        out_s = OUT_DIR / "forecast_week_hourly_scenarios.csv"
        fc_s.to_csv(out_s, index=False)
        print("OK ->", out_s)

    print("\nListo.")
    print("Consejo: si el resto de la semana se te va, reduce aún más ANCHOR_WEIGHTS (p.ej. [1,0.3,0,0,0,0,0]).")
