"""Carga de resultados crudos y agregación calidad/costo por configuración.

Es el paso que convierte las filas por documento que escribe el runner en la
tabla que responde la pregunta del objetivo específico 6: qué configuración
ofrece el mejor compromiso entre calidad y costo.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from resumidor.eval.metrics import bertscore, rouge

COLUMNAS_CALIDAD = ["rouge1", "rouge2", "rougeLsum", "bertscore"]
COLUMNAS_COSTO = ["latencia_s", "pico_memoria_mb", "ratio_compresion"]


def cargar_resultados(directorio: str | pathlib.Path) -> pd.DataFrame:
    """Lee todos los `.jsonl` de resultados en un solo DataFrame."""
    directorio = pathlib.Path(directorio)
    filas = [
        json.loads(linea)
        for archivo in sorted(directorio.glob("*.jsonl"))
        for linea in archivo.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    if not filas:
        raise FileNotFoundError(
            f"No hay resultados en {directorio}. Ejecuta primero "
            "`scripts/run_experiment.py` con alguna configuración."
        )
    return pd.DataFrame(filas)


def calificar(df: pd.DataFrame, *, con_bertscore: bool = True) -> pd.DataFrame:
    """Añade las columnas de calidad a los resultados crudos.

    ROUGE se calcula fila a fila; BERTScore en un solo lote por el costo de
    cargar su modelo.
    """
    faltan = {"resumen_generado", "resumen_referencia"} - set(df.columns)
    if faltan:
        raise KeyError(f"Faltan columnas en los resultados: {sorted(faltan)}")

    df = df.copy()
    puntajes = [
        rouge(fila.resumen_generado, fila.resumen_referencia)
        for fila in df.itertuples()
    ]
    df["rouge1"] = [p.rouge1 for p in puntajes]
    df["rouge2"] = [p.rouge2 for p in puntajes]
    df["rougeLsum"] = [p.rougeLsum for p in puntajes]

    if con_bertscore:
        df["bertscore"] = bertscore(
            df["resumen_generado"].tolist(), df["resumen_referencia"].tolist()
        )
    return df


def resumir_por_configuracion(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla calidad/costo, una fila por configuración.

    La latencia se reporta como **p50 y p95**, no como media: el objetivo
    específico 5 los pide así, y la media es engañosa en una distribución con
    cola larga como la de estos documentos.
    """
    calidad = [c for c in COLUMNAS_CALIDAD if c in df.columns]

    agregados: dict = {c: "mean" for c in calidad}
    agregados["pico_memoria_mb"] = "max"
    agregados["ratio_compresion"] = "mean"
    agregados["tokens_salida"] = "mean"
    agregados["invocaciones_modelo"] = "mean"
    agregados["doc_id"] = "count"

    tabla = df.groupby(["config_id", "modelo", "estrategia"]).agg(agregados)
    latencias = df.groupby(["config_id", "modelo", "estrategia"])["latencia_s"]
    tabla["latencia_p50"] = latencias.quantile(0.50)
    tabla["latencia_p95"] = latencias.quantile(0.95)

    tabla = tabla.rename(
        columns={
            "doc_id": "n",
            "tokens_salida": "tokens_generados",
            "invocaciones_modelo": "invocaciones",
            "pico_memoria_mb": "memoria_pico_mb",
        }
    )
    orden = [
        "n",
        *calidad,
        "latencia_p50",
        "latencia_p95",
        "memoria_pico_mb",
        "ratio_compresion",
        "tokens_generados",
        "invocaciones",
    ]
    return tabla[orden].sort_values(
        calidad[0] if calidad else "latencia_p50", ascending=False
    )
