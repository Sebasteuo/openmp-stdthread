#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE-4302 — Taller 01
Script: plot.py

Qué hace (en corto):
  1) Leemos un CSV (por defecto: results/raw/last.csv) con columnas:
     backend, variant, threads, N, bins, min, max, seed, gen_ms, hist_ms, total_ms, sum_hist
  2) Agregamos por (backend, variant, threads) usando mediana (o media) de la métrica elegida.
  3) Calculamos speedup (Tref/Tn) y eficiencia (speedup/n) por par (backend, variant).
  4) Generamos gráficos:
      - results/time_vs_threads.png
      - results/speedup_vs_threads.png
      - results/efficiency_vs_threads.png
     y además versiones etiquetadas con la métrica:
      - results/time_vs_threads_<metric>.png, etc.

Uso rápido:
  python3 scripts/plot.py results/raw/last.csv
  python3 scripts/plot.py results/raw/last.csv --metric total_ms --agg median
  python3 scripts/plot.py results/raw/last.csv --filter-backends openmp threads --filter-variants private atomic
  python3 scripts/plot.py --help

Requisitos:
  - pandas, matplotlib

Notas:
  - No forzamos estilos ni colores (compatibles con la rúbrica).
  - Si no existe el punto con 1 hilo para un (backend, variant), normalizamos con el menor #hilos que haya.
  - Si se detecta speedup > #hilos, mostramos una advertencia (suele pasar con N chico/VM/redondeo de ms).
"""

import os
import sys
import argparse
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Columnas mínimas que esperamos
REQUIRED_COLS = [
    "backend","variant","threads","N","bins","min","max","seed",
    "gen_ms","hist_ms","total_ms","sum_hist"
]

# ----------------------------------------------------------------------
# CLI (opciones de entrada)
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parseamos argumentos de línea de comandos:
      - csv de entrada
      - carpeta de salida
      - métrica a usar (total_ms | gen_ms | hist_ms)
      - función de agregación (median | mean)
      - filtros para backend/variant
      - nivel de verbosidad
    """
    p = argparse.ArgumentParser(
        description="Agrega CSV de benchmarks y genera gráficos (tiempo, speedup, eficiencia)."
    )
    p.add_argument("csv", nargs="?", default="results/raw/last.csv",
                   help="Ruta del CSV de entrada (por defecto: results/raw/last.csv).")
    p.add_argument("--outdir", default="results",
                   help="Carpeta de salida para CSVs y PNGs (default: results).")
    p.add_argument("--metric", default="total_ms",
                   choices=["total_ms","gen_ms","hist_ms"],
                   help="Métrica de tiempo a graficar (default: total_ms).")
    p.add_argument("--agg", default="median", choices=["median","mean"],
                   help="Agregación por grupo (default: median).")
    p.add_argument("--filter-backends", nargs="*", default=None,
                   help="Filtra backends (ej: openmp threads).")
    p.add_argument("--filter-variants", nargs="*", default=None,
                   help="Filtra variants (ej: private atomic mutex).")
    p.add_argument("--quiet", action="store_true",
                   help="Menos mensajes por consola.")
    return p.parse_args()

# Log sencillito para mensajes de estado
def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)

# ----------------------------------------------------------------------
# Carga y validaciones básicas del CSV
# ----------------------------------------------------------------------
def load_and_validate(csv_path: str, quiet: bool=False) -> pd.DataFrame:
    """
    Cargamos el CSV, revisamos que tenga las columnas mínimas, convertimos
    tipos donde toca y limpiamos filas con NaN en campos clave.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe el archivo CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # ¿Faltan columnas?
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    # Aseguramos tipos numéricos donde importan
    df["threads"] = pd.to_numeric(df["threads"], errors="coerce").astype("Int64")
    for col in ["gen_ms","hist_ms","total_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtramos filas con NaN en columnas clave
    before = len(df)
    df = df.dropna(subset=["backend","variant","threads","gen_ms","hist_ms","total_ms"])
    after = len(df)
    if before != after:
        log(f"[i] Filtradas {before-after} filas inválidas (NaN en columnas clave).", quiet)

    # 'threads' como int normal para ordenar sin drama
    df["threads"] = df["threads"].astype(int)
    return df

# ----------------------------------------------------------------------
# Filtros opcionales por backend/variant
# ----------------------------------------------------------------------
def apply_filters(df: pd.DataFrame,
                  backends: List[str] = None,
                  variants: List[str] = None,
                  quiet: bool=False) -> pd.DataFrame:
    """
    Si nos pasan filtros, dejamos sólo las filas que coinciden.
    Si tras filtrar no queda nada, avisamos con error.
    """
    if backends:
        df = df[df["backend"].isin(backends)]
        log(f"[i] Filtrado por backends: {backends}", quiet)
    if variants:
        df = df[df["variant"].isin(variants)]
        log(f"[i] Filtrado por variants: {variants}", quiet)
    if df.empty:
        raise ValueError("El DataFrame quedó vacío tras aplicar filtros.")
    return df

# ----------------------------------------------------------------------
# Agregado + speedup/eficiencia
# ----------------------------------------------------------------------
def aggregate(df: pd.DataFrame, metric: str, agg: str, outdir: str, quiet: bool=False) -> pd.DataFrame:
    """
    Agregamos por (backend, variant, threads) con la métrica elegida,
    guardamos el agregado base y, además, calculamos speedup y eficiencia.
    """
    os.makedirs(outdir, exist_ok=True)

    # Usamos una copia y renombramos la métrica elegida a 'time_ms' (para simplificar)
    df = df.copy()
    df["time_ms"] = df[metric]

    # Agregación (mediana por defecto; media si la piden)
    if agg == "median":
        aggdf = (df.groupby(["backend","variant","threads"], as_index=False)["time_ms"]
                   .median())
    else:
        aggdf = (df.groupby(["backend","variant","threads"], as_index=False)["time_ms"]
                   .mean())

    aggdf = aggdf.sort_values(["backend","variant","threads"]).reset_index(drop=True)

    # Guardamos el agregado base
    base_csv = os.path.join(outdir, "aggregated.csv")
    aggdf.to_csv(base_csv, index=False)
    log(f"[✓] Guardado agregado base: {base_csv}", quiet)

    # Speedup/eficiencia por cada (backend, variant)
    rows = []
    for (b, v), sub in aggdf.groupby(["backend","variant"]):
        sub = sub.sort_values("threads").copy()

        # Referencia: tiempo con 1 hilo si existe; si no, el menor #hilos disponible
        if 1 in set(sub["threads"]):
            t_ref = float(sub.loc[sub["threads"]==1, "time_ms"].iloc[0])
            ref_threads = 1
        else:
            t_ref = float(sub.iloc[0]["time_ms"])
            ref_threads = int(sub.iloc[0]["threads"])
            log(f"[!] {b}-{v}: no hay punto con 1 hilo; usamos {ref_threads} como referencia.", quiet)

        if t_ref <= 0:
            # Caso borde raro (no debería pasar): dejamos NaN y avisamos
            log(f"[!] {b}-{v}: tiempo de referencia no positivo ({t_ref}); omitimos speedup/eficiencia.", quiet)
            sub["speedup"] = float("nan")
            sub["efficiency"] = float("nan")
        else:
            sub["speedup"]   = t_ref / sub["time_ms"]
            sub["efficiency"] = sub["speedup"] / sub["threads"]

        rows.append(sub)

    out = pd.concat(rows, ignore_index=True)

    # Guardamos el extendido con speedup/eficiencia
    out_csv = os.path.join(outdir, "aggregated_with_speedup.csv")
    out.to_csv(out_csv, index=False)
    log(f"[✓] Guardado agregado con speedup/eficiencia: {out_csv}", quiet)
    return out

# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def plot_curves(agg: pd.DataFrame, metric: str, outdir: str, quiet: bool=False) -> Dict[str, str]:
    """
    Graficamos tres figuras: tiempo, speedup y eficiencia.
    Devolvemos rutas a los PNG generados.
    """
    paths: Dict[str, str] = {}

    # --- Tiempo vs hilos ---
    plt.figure()
    for (backend, variant), sub in agg.groupby(["backend","variant"]):
        sub = sub.sort_values("threads")
        plt.plot(sub["threads"], sub["time_ms"], marker="o", label=f"{backend}-{variant}")
    plt.xlabel("Hilos")
    plt.ylabel("Tiempo (ms)")
    plt.title("Tiempo vs #hilos")
    plt.grid(True)
    plt.legend()
    p1 = os.path.join(outdir, "time_vs_threads.png")
    plt.savefig(p1, bbox_inches="tight")
    paths["time"] = p1

    # Versión con el nombre de la métrica (para dejar claro qué se graficó)
    p1m = os.path.join(outdir, f"time_vs_threads_{metric}.png")
    plt.savefig(p1m, bbox_inches="tight")
    paths["time_metric"] = p1m

    # --- Speedup ---
    plt.figure()
    for (backend, variant), sub in agg.groupby(["backend","variant"]):
        sub = sub.sort_values("threads")
        if "speedup" not in sub or sub["speedup"].isna().all():
            continue
        plt.plot(sub["threads"], sub["speedup"], marker="o", label=f"{backend}-{variant}")
    plt.xlabel("Hilos")
    plt.ylabel("Speedup (Tref/Tn)")
    plt.title("Speedup vs #hilos")
    plt.grid(True)
    plt.legend()
    p2 = os.path.join(outdir, "speedup_vs_threads.png")
    plt.savefig(p2, bbox_inches="tight")
    paths["speedup"] = p2

    p2m = os.path.join(outdir, f"speedup_vs_threads_{metric}.png")
    plt.savefig(p2m, bbox_inches="tight")
    paths["speedup_metric"] = p2m

    # --- Eficiencia ---
    plt.figure()
    for (backend, variant), sub in agg.groupby(["backend","variant"]):
        sub = sub.sort_values("threads")
        if "efficiency" not in sub or sub["efficiency"].isna().all():
            continue
        plt.plot(sub["threads"], sub["efficiency"], marker="o", label=f"{backend}-{variant}")
    plt.xlabel("Hilos")
    plt.ylabel("Eficiencia (speedup/n)")
    plt.title("Eficiencia vs #hilos")
    plt.grid(True)
    plt.legend()
    p3 = os.path.join(outdir, "efficiency_vs_threads.png")
    plt.savefig(p3, bbox_inches="tight")
    paths["efficiency"] = p3

    p3m = os.path.join(outdir, f"efficiency_vs_threads_{metric}.png")
    plt.savefig(p3m, bbox_inches="tight")
    paths["efficiency_metric"] = p3m

    log(f"[✓] Gráficos listos en {outdir}/*.png", quiet)
    return paths

# ----------------------------------------------------------------------
# Aviso si hay “speedup superlineal” (para no confundir al lector)
# ----------------------------------------------------------------------
def warn_superlinear(agg: pd.DataFrame, quiet: bool=False) -> None:
    if "speedup" not in agg.columns or "threads" not in agg.columns:
        return
    sup = agg[(agg["speedup"].notna()) & (agg["speedup"] > agg["threads"])]
    if not sup.empty:
        log("\n[!] Ojo: hay casos con speedup > #hilos (superlineal).", quiet)
        for _, row in sup.iterrows():
            log(f"    - {row['backend']}-{row['variant']} @ {int(row['threads'])} hilos: "
                f"speedup={row['speedup']:.2f}", quiet)
        log("    Sugerencia: usar N grande y Linux nativo para mediciones finales.\n", quiet)

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_validate(args.csv, quiet=args.quiet)
    df = apply_filters(df, args.filter_backends, args.filter_variants, quiet=args.quiet)

    # Armamos el agregado y el extendido (con speedup/eficiencia)
    agg_full = aggregate(df, args.metric, args.agg, args.outdir, quiet=args.quiet)

    # Graficamos usando la métrica elegida (para nombrar los PNG *_<metric>.png)
    plot_curves(agg_full, args.metric, args.outdir, quiet=args.quiet)
    warn_superlinear(agg_full, quiet=args.quiet)

    print(f"Tabla agregada: {os.path.join(args.outdir,'aggregated.csv')}")
    print(f"Tabla agregada con speedup: {os.path.join(args.outdir,'aggregated_with_speedup.csv')}")

if __name__ == "__main__":
    main()
