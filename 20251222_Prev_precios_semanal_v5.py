import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG (CAMBIA SOLO ESTO)
# =========================
DATA_DIR_HIST = "precios"      # carpeta con 2018.csv, 2019.csv, ...
TOMORROW_DATE = "2025-12-23"         # YYYY-MM-DD
TOMORROW_MEAN_EUR_MWH = 140.55        # media diaria conocida de mañana (€/MWh)
OUTFILE = "forecast_week_hourly_FINAL.csv"
DAYS = 7

# Parámetros finales recomendados
LOW_Q = 0.05                         # P5 de medias diarias por clase (suelo dinámico)
VALLEY_HOURS = {1, 2, 3, 4}          # horas valle profundo (0-23)
VALLEY_MULT = 0.50                   # multiplicador valle (más bajo => más valle barato)
LAMBDAS_7D = [1.00, 0.80, 0.65, 0.50, 0.40, 0.30, 0.25]  # anclaje decreciente D+1..D+7

# =========================
# CLASES / CALENDARIO
# =========================
FESTIVOS_FIJOS_ES = {
    (1, 1), (1, 6), (5, 1), (8, 15),
    (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
}

def day_type(dow: int, holiday: bool) -> str:
    if holiday or dow == 6: return "DOM"
    if dow == 5: return "SAT"
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

# =========================
# CARGA HISTÓRICO (Year,Month,Day,Hour(1-24),Value €/MWh)
# =========================
def load_history_csvs(data_dir: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("20*.csv"))
    if not files:
        raise FileNotFoundError(f"No encuentro CSVs 20*.csv en: {data_dir.resolve()}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    df = df.rename(columns={
        "Year": "year", "Month": "month", "Day": "day", "Hour": "hour_124", "Value": "price_eur_mwh"
    })
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour_124"] = df["hour_124"].astype(int)
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    # 1-24 -> 0-23
    df["hour"] = df["hour_124"] - 1
    if df["hour"].min() < 0 or df["hour"].max() > 23:
        raise ValueError("Horas fuera de 0-23 tras convertir Hour 1-24")

    df = add_calendar_columns(df)
    df = df.dropna(subset=["price_eur_mwh"])
    return df

# =========================
# α por clase y hora (shape)
# =========================
def build_alpha_profiles(df_hist: pd.DataFrame, q=(0.05, 0.95)) -> pd.DataFrame:
    df = df_hist.copy()
    daily_mean = df.groupby("date")["price_eur_mwh"].mean().rename("p_mean_day")
    df = df.merge(daily_mean, on="date", how="left")
    df["ratio"] = df["price_eur_mwh"] / df["p_mean_day"]

    lo_q, hi_q = q
    def clip_group(g):
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

# =========================
# µ_c y suelo dinámico (P5 de media diaria por clase)
# =========================
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

# =========================
# Factor por tipo de día (calibrado)
# =========================
def calibrate_type_factor(df_hist: pd.DataFrame) -> dict:
    daily = (df_hist.groupby(["date","type"])["price_eur_mwh"]
                    .mean()
                    .reset_index()
                    .rename(columns={"price_eur_mwh":"p_mean_day"}))
    base = daily.loc[daily["type"]=="LAB", "p_mean_day"].mean()
    out = {}
    for t in ["LAB","SAT","DOM"]:
        m = daily.loc[daily["type"]==t, "p_mean_day"].mean()
        out[t] = float(m / base) if base > 0 else 1.0
    return out

# =========================
# Valle + renormalización para mantener la media diaria
# =========================
def valley_multiplier(hour: int) -> float:
    return VALLEY_MULT if hour in VALLEY_HOURS else 1.0

def apply_valley_and_renormalize(day_df: pd.DataFrame) -> pd.DataFrame:
    df = day_df.copy()
    df["m_valley"] = df["hour"].apply(valley_multiplier)
    df["alpha_adj"] = df["alpha"] * df["m_valley"]

    # Renormaliza: media(alpha_adj)=1 para no alterar la media diaria prevista
    mean_adj = df["alpha_adj"].mean()
    if mean_adj <= 0:
        mean_adj = 1.0
    df["alpha_adj"] = df["alpha_adj"] / mean_adj

    df["Price_EUR_MWh"] = df["p_mean_pred_eur_mwh"] * df["alpha_adj"]
    return df

# =========================
# Forecast semanal desde media diaria de mañana
# =========================
def forecast_week_from_tomorrow_mean(df_hist: pd.DataFrame,
                                     tomorrow_date: str,
                                     tomorrow_mean_eur_mwh: float,
                                     days: int = 7,
                                     low_q: float = 0.05) -> pd.DataFrame:

    alpha = build_alpha_profiles(df_hist)
    stats = build_class_daily_stats(df_hist).set_index("class")
    mu_global = stats["mu"].mean()

    qlow_df = build_class_daily_low_quantiles(df_hist, q=low_q).set_index("class")
    p_low_global = float(qlow_df["p_low"].median()) if len(qlow_df) else 0.0

    type_factor = calibrate_type_factor(df_hist)

    start = pd.to_datetime(tomorrow_date).normalize()
    dates = [start + pd.Timedelta(days=i) for i in range(days)]
    cal = pd.DataFrame({"date": dates})
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["day"] = cal["date"].dt.day
    cal = add_calendar_columns(cal)

    # ancla (solo media diaria)
    anchor_class = cal.loc[cal["date"] == start, "class"].iloc[0]
    mu_anchor = float(stats.loc[anchor_class, "mu"]) if anchor_class in stats.index else float(mu_global)
    k = float(tomorrow_mean_eur_mwh / mu_anchor) if mu_anchor > 0 else 1.0

    # medias diarias previstas (con suelo dinámico)
    p_means = []
    for i, row in cal.iterrows():
        c = row["class"]
        t = row["type"]

        mu_c = float(stats.loc[c, "mu"]) if c in stats.index else float(mu_global)
        lam = LAMBDAS_7D[i]
        kd = k ** lam
        ft = float(type_factor.get(t, 1.0))

        p_pred = mu_c * kd * ft

        # suelo P5 por clase
        p_low_c = float(qlow_df.loc[c, "p_low"]) if c in qlow_df.index else p_low_global
        p_pred = max(p_pred, p_low_c)

        p_means.append(float(p_pred))

    daily_fc = cal[["date","class","type"]].copy()
    daily_fc["p_mean_pred_eur_mwh"] = p_means
    daily_fc["k_anchor"] = k
    daily_fc["mu_anchor"] = mu_anchor

    # desagregar a horas con α
    out = daily_fc.merge(alpha, on="class", how="left")
    if out["alpha"].isna().any():
        missing = out.loc[out["alpha"].isna(), "class"].unique()
        raise ValueError(f"Faltan alphas para clases: {missing}")

    # valle + renormalización por día
    out = (out.sort_values(["date","hour"])
              .groupby("date", group_keys=False)
              .apply(apply_valley_and_renormalize))

    out = out.sort_values(["date","hour"]).reset_index(drop=True)
    out["Source"] = "FORECAST_FINAL"
    out["TomorrowMean_EUR_MWh"] = float(tomorrow_mean_eur_mwh)
    out["LowQuantileUsed"] = float(low_q)
    out["ValleyHours"] = str(sorted(list(VALLEY_HOURS)))
    out["ValleyMult"] = float(VALLEY_MULT)

    return out[[
        "date","hour","class","type",
        "p_mean_pred_eur_mwh","Price_EUR_MWh",
        "k_anchor","mu_anchor",
        "Source","TomorrowMean_EUR_MWh","LowQuantileUsed","ValleyHours","ValleyMult"
    ]]

# =========================
# MAIN (EJECUTA Y GUARDA)
# =========================
if __name__ == "__main__":
    df_hist = load_history_csvs(DATA_DIR_HIST)

    week = forecast_week_from_tomorrow_mean(
        df_hist=df_hist,
        tomorrow_date=TOMORROW_DATE,
        tomorrow_mean_eur_mwh=TOMORROW_MEAN_EUR_MWH,
        days=DAYS,
        low_q=LOW_Q
    )

    print("OK. Filas generadas:", len(week))
    print(week.head(12))

    # chequeo: la media de mañana debe cuadrar
    tomorrow = week[week["date"] == pd.to_datetime(TOMORROW_DATE)]
    print("Media mañana (pred):", tomorrow["Price_EUR_MWh"].mean(), " vs ", TOMORROW_MEAN_EUR_MWH)

    week.to_csv(OUTFILE, index=False)
    print("Guardado:", OUTFILE)
