#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Previsión horaria para "mañana" cuando SOLO conoces la media diaria (€/MWh).

Entrada:
- CSVs anuales en una carpeta (por defecto ./precios):
  2018.csv, 2019.csv, ..., 2025.csv
  con columnas: Year, Month, Day, Hour, Value
  Hour: 1..24
  Value: €/MWh

Salida (en ./out):
- alpha_profiles_12classes.csv     (12 clases x 24 horas)
- class_daily_stats.csv            (mu, p10, p50, p90, n_days por clase)
- tomorrow_hourly_forecast.csv     (24 horas estimadas para mañana)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN (EDITA AQUÍ)
# =========================================================

# 1) Media diaria conocida para mañana (€/MWh). Pon tu valor aquí:
TOMORROW_DAILY_MEAN_EUR_MWH = 140.55  # <-- CAMBIA ESTO

# 2) Carpeta con los CSVs anuales
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "precios"   # <-- mete aquí 2018.csv, 2019.csv, ...

# 3) Carpeta de salida
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

# 4) Festivos nacionales fijos (mínimo)
FESTIVOS_FIJOS = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

# 5) Si quieres tratar la hora como 0..23 internamente (recomendado)
INTERNAL_HOUR_0_23 = True

# =========================================================
# HELPERS
# =========================================================

def season_from_month(m: int) -> str:
    """Estación meteorológica."""
    if m in (12, 1, 2): return "INV"
    if m in (3, 4, 5):  return "PRI"
    if m in (6, 7, 8):  return "VER"
    return "OTO"

def day_class(ts: pd.Timestamp) -> str:
    """Clase: LAB/SAT/DOM x INV/PRI/VER/OTO => 12 clases."""
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

    s = season_from_month(m)
    return f"{t}_{s}"

def read_one_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV tolerando separadores comunes."""
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

    # Tipos
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour_124"] = df["hour_124"].astype(int)

    # Value puede venir como texto con coma decimal
    if df["price_eur_mwh"].dtype == object:
        df["price_eur_mwh"] = df["price_eur_mwh"].astype(str).str.replace(",", ".", regex=False)

    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    # Fecha
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")

    # Hora
    if INTERNAL_HOUR_0_23:
        df["hour"] = df["hour_124"] - 1  # 0..23
        if df["hour"].min() < 0 or df["hour"].max() > 23:
            raise ValueError("Horas fuera de rango al convertir 1..24 a 0..23.")
    else:
        df["hour"] = df["hour_124"]      # 1..24
        if df["hour"].min() < 1 or df["hour"].max() > 24:
            raise ValueError("Horas fuera de rango (esperaba 1..24).")

    # Limpieza
    df = df.dropna(subset=["date", "price_eur_mwh"])
    return df

def compute_alpha_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    alpha_{c,h} = mean_t( p_{t,h}/mean_h(p_{t,h}) ) para t en clase c
    + recorte percentil P5-P95 por (c,h) para evitar extremos espurios.
    """
    tmp = df.copy()
    tmp["class"] = tmp["date"].apply(day_class)

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
    """Estadísticos de la media diaria por clase (para documentación/escenarios)."""
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

def forecast_tomorrow_hourly(daily_mean_eur_mwh: float,
                             tomorrow_date: pd.Timestamp,
                             alpha: pd.DataFrame) -> pd.DataFrame:
    """
    Si solo conocemos la media diaria de mañana:
      p_hat(h) = mean_day * alpha_{class(tomorrow),h}
    """
    c = day_class(tomorrow_date)
    a = alpha[alpha["class"] == c].sort_values("hour")

    if len(a) != 24:
        # Backoff: perfil global por hora si falta algún dato de clase
        a = alpha.groupby("hour", as_index=False)["alpha"].mean().sort_values("hour")

    prices = daily_mean_eur_mwh * a["alpha"].to_numpy()

    out = pd.DataFrame({
        "Date": [tomorrow_date.date().isoformat()] * 24,
        "Hour": a["hour"].to_numpy(),
        "Price_EUR_MWh": prices
    })

    # Si quieres devolver Hour 1..24 en salida
    if INTERNAL_HOUR_0_23:
        out["Hour"] = out["Hour"].astype(int) + 1  # vuelve a 1..24 para comodidad

    return out

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Leyendo histórico desde:", DATA_DIR.resolve())
    df = build_master_dataframe(DATA_DIR)

    # Construcción perfiles y stats
    alpha = compute_alpha_profiles(df)
    stats = compute_class_daily_stats(df)

    alpha_path = OUT_DIR / "alpha_profiles_12classes.csv"
    stats_path = OUT_DIR / "class_daily_stats.csv"
    alpha.to_csv(alpha_path, index=False)
    stats.to_csv(stats_path, index=False)

    print("OK ->", alpha_path)
    print("OK ->", stats_path)

    # Mañana
    tomorrow = pd.Timestamp((datetime.now().date() + timedelta(days=1)).isoformat())
    print("Mañana (fecha):", tomorrow.date().isoformat(), "| Clase:", day_class(tomorrow))
    print("Media diaria usada (€/MWh):", TOMORROW_DAILY_MEAN_EUR_MWH)

    forecast = forecast_tomorrow_hourly(TOMORROW_DAILY_MEAN_EUR_MWH, tomorrow, alpha)
    out_path = OUT_DIR / "tomorrow_hourly_forecast.csv"
    forecast.to_csv(out_path, index=False)

    print("OK ->", out_path)
    print("Listo. (Salida con Hour en 1..24)")
