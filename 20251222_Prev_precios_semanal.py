#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Previsión horaria semanal cuando SOLO conoces la media diaria de mañana (y opcionalmente pasado).

Entrada:
- CSVs anuales en ./precios: 2018.csv ... 2025.csv
  Columnas: Year, Month, Day, Hour (1..24), Value (€/MWh)

Idea:
1) Aprendemos del histórico:
   - alpha_{class,h}: perfil horario normalizado por clase (12 clases = tipo_día x estación)
   - mu_class: nivel típico (media) de la media diaria por clase
2) Operación:
   - Conoces la media diaria de mañana (ancla). (Opcional: también la de pasado)
   - Estimas media diaria D+2..D+7 con baseline mu_class ajustado por el ancla (escala con decaimiento)
   - Desagregas a horas con alpha_{class,h}

Salida (en ./out):
- alpha_profiles_12classes.csv
- class_daily_stats.csv
- forecast_week_hourly.csv  (Date, Hour(1..24), Price_EUR_MWh, Source)
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
# - Mañana: OBLIGATORIA
# - Pasado: OPCIONAL (si no la tienes, deja None)
TOMORROW_DAILY_MEAN_EUR_MWH = 140.55   # <-- CAMBIA ESTO
DAY_AFTER_DAILY_MEAN_EUR_MWH = None   # <-- o pon un float si lo tienes

# Horizonte a generar (días): 7 => mañana + 6 días más
HORIZON_DAYS = 7

# Carpeta con CSVs anuales
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "precios"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

# Festivos nacionales fijos (mínimo)
FESTIVOS_FIJOS = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

# Horas internas 0..23 (recomendado); salida en 1..24
INTERNAL_HOUR_0_23 = True

# Escalado del nivel (ancla) y decaimiento con horizonte
S_MIN, S_MAX = 0.6, 1.6      # límites para no exagerar
K_DECAY = 0.3                # 0 => sin decaimiento; 0.3 => decae moderado

# Si tienes mañana y pasado, puedes estimar "tendencia" del nivel (suave)
USE_TREND_IF_TWO_ANCHORS = True
TREND_CAP_PER_DAY = 0.15     # máximo +/-15% por día de cambio en escala (prudente)

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

def _alpha_for_class_or_global(alpha: pd.DataFrame, c: str) -> pd.DataFrame:
    a = alpha[alpha["class"] == c].sort_values("hour")
    if len(a) == 24:
        return a
    # backoff global por hora
    return alpha.groupby("hour", as_index=False)["alpha"].mean().sort_values("hour")

def _mu_for_class(class_stats: pd.DataFrame, c: str) -> float:
    row = class_stats.loc[class_stats["class"] == c, "mu"]
    if row.empty:
        raise ValueError(f"No hay mu para la clase {c}. ¿Histórico insuficiente?")
    return float(row.iloc[0])

def _scale_series(tomorrow_date: pd.Timestamp,
                  p_mean_tom: float,
                  class_stats: pd.DataFrame,
                  p_mean_day_after: float | None,
                  k_decay: float = K_DECAY) -> dict[int, float]:
    """
    Devuelve s_i para i=0..HORIZON_DAYS-1, donde:
      i=0 mañana
      i=1 pasado mañana
    s_i escala el baseline mu_{class(d+i)}.

    Caso 1 ancla:
      s_i = clip(s0 ** exp(-k*i))
    Caso 2 anclas (opcional):
      estima tendencia en log-escala y la aplica suavemente, con cap por día.
    """
    c0 = day_class(tomorrow_date)
    mu0 = _mu_for_class(class_stats, c0)
    s0 = (p_mean_tom / mu0) if mu0 > 0 else 1.0
    s0 = float(np.clip(s0, S_MIN, S_MAX))

    if p_mean_day_after is None or not USE_TREND_IF_TWO_ANCHORS:
        # Solo 1 ancla
        return {i: float(np.clip(s0 ** np.exp(-k_decay * i), S_MIN, S_MAX)) for i in range(HORIZON_DAYS)}

    # Dos anclas: mañana y pasado (i=0,1)
    d1 = tomorrow_date + pd.Timedelta(days=1)
    c1 = day_class(d1)
    mu1 = _mu_for_class(class_stats, c1)
    s1 = (p_mean_day_after / mu1) if mu1 > 0 else s0
    s1 = float(np.clip(s1, S_MIN, S_MAX))

    # Tendencia en log(s) por día, limitada
    log_s0 = np.log(s0)
    log_s1 = np.log(s1)
    delta = log_s1 - log_s0  # cambio de log escala por 1 día
    delta = float(np.clip(delta, -TREND_CAP_PER_DAY, TREND_CAP_PER_DAY))

    s = {}
    for i in range(HORIZON_DAYS):
        # aplica tendencia (lineal en log) y además decaimiento de la "confianza" con horizonte
        # conf=1 hasta i=1, luego decae
        if i <= 1:
            log_si = log_s0 + delta * i
        else:
            conf = float(np.exp(-k_decay * (i - 1)))
            log_si = (log_s0 + delta * i) * conf + log_s0 * (1 - conf)

        si = float(np.clip(np.exp(log_si), S_MIN, S_MAX))
        s[i] = si

    return s

def forecast_week_hourly(tomorrow_date: pd.Timestamp,
                         p_mean_tom: float,
                         alpha: pd.DataFrame,
                         class_stats: pd.DataFrame,
                         p_mean_day_after: float | None,
                         days: int = HORIZON_DAYS) -> pd.DataFrame:
    """
    Genera forecast horario para días=HORIZON_DAYS:
    - i=0 mañana: media diaria conocida p_mean_tom
    - i=1 pasado: si media conocida, se respeta; si no, se estima con baseline escalado
    - i>=2: se estima con baseline escalado y (opcional) decaimiento
    """
    # escala temporal (en función de anclas)
    s_series = _scale_series(tomorrow_date, p_mean_tom, class_stats, p_mean_day_after)

    rows = []
    for i in range(days):
        d = tomorrow_date + pd.Timedelta(days=i)
        c = day_class(d)
        mu = _mu_for_class(class_stats, c)

        # Media diaria objetivo
        if i == 0:
            p_mean_day = float(p_mean_tom)
            source = "ANCHOR_TOMORROW_MEAN"
        elif i == 1 and p_mean_day_after is not None:
            p_mean_day = float(p_mean_day_after)
            source = "ANCHOR_DAY_AFTER_MEAN"
        else:
            p_mean_day = float(mu * s_series[i])
            source = "ESTIMATED_MEAN"

        # Perfil horario
        a = _alpha_for_class_or_global(alpha, c)
        prices = p_mean_day * a["alpha"].to_numpy()

        for hour, price in zip(a["hour"].to_numpy(), prices):
            out_hour = int(hour) + 1 if INTERNAL_HOUR_0_23 else int(hour)
            rows.append({
                "Date": d.date().isoformat(),
                "Hour": out_hour,                 # 1..24
                "Price_EUR_MWh": float(price),
                "Source": source,
                "Class": c
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

    # Fechas del horizonte (mañana como inicio)
    tomorrow = pd.Timestamp((datetime.now().date() + timedelta(days=1)).isoformat())

    print("\n--- ANCLAS ---")
    print("Mañana:", tomorrow.date().isoformat(), "| Clase:", day_class(tomorrow), "| Media diaria:", TOMORROW_DAILY_MEAN_EUR_MWH, "€/MWh")
    if DAY_AFTER_DAILY_MEAN_EUR_MWH is not None:
        d1 = tomorrow + pd.Timedelta(days=1)
        print("Pasado:", d1.date().isoformat(), "| Clase:", day_class(d1), "| Media diaria:", DAY_AFTER_DAILY_MEAN_EUR_MWH, "€/MWh")
    else:
        print("Pasado: (sin media diaria)")

    forecast = forecast_week_hourly(
        tomorrow_date=tomorrow,
        p_mean_tom=float(TOMORROW_DAILY_MEAN_EUR_MWH),
        alpha=alpha,
        class_stats=class_stats,
        p_mean_day_after=DAY_AFTER_DAILY_MEAN_EUR_MWH,
        days=HORIZON_DAYS
    )

    out_path = OUT_DIR / "forecast_week_hourly.csv"
    forecast.to_csv(out_path, index=False)

    print("\nOK ->", out_path)
    print("Listo. (Salida: Date, Hour(1..24), Price_EUR_MWh, Source, Class)")
