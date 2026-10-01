"""Califica los resultados crudos y produce las tablas del análisis.

Uso:
    uv run python scripts/evaluar.py                      # todo, con BERTScore
    uv run python scripts/evaluar.py --sin-bertscore      # rápido, solo ROUGE
    uv run python scripts/evaluar.py --resultados /ruta/a/jsonl --salida /ruta

Lee todos los `experiments/results/*.jsonl` que escribe `run_experiment.py`
—uno por configuración— y escribe en `--salida`:

    calificado.csv    una fila por (configuración, documento) con ROUGE y
                      BERTScore; es la base de todo lo demás
    resumen.csv       calidad y costo por configuración (latencia p50/p95)
    por_estrato.csv   lo mismo, desglosado por estrato de longitud
    pareadas.csv      pruebas de Wilcoxon pareadas, IC 95 % y Holm
    recuperacion.csv  % del ROUGE-Lsum, latencia y memoria de LED (ADR-003)

BERTScore descarga `roberta-large` (~1,4 GB) la primera vez y es la parte
lenta: en CPU, unos minutos por cada mil resúmenes; en GPU, segundos.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from resumidor.eval import (
    calificar,
    cargar_resultados,
    pruebas_pareadas,
    recuperacion_frente_a,
    resumir_por_configuracion,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resultados", type=pathlib.Path, default=pathlib.Path("experiments/results")
    )
    p.add_argument("--salida", type=pathlib.Path, default=None)
    p.add_argument("--sin-bertscore", action="store_true")
    p.add_argument(
        "--baseline", default="baseline_led", help="config_id del techo (ADR-003)"
    )
    p.add_argument(
        "--referencias",
        nargs="*",
        default=["baseline_led", "control_longt5_sin_afinar"],
        help="configuraciones que pueden tener menos documentos que el factorial",
    )
    args = p.parse_args()
    salida = args.salida or args.resultados / "analisis"
    salida.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    df = cargar_resultados(args.resultados)
    print(f"{len(df)} filas · {df['config_id'].nunique()} configuraciones\n")
    print(df.groupby("config_id").size().rename("documentos").to_string(), "\n")

    # El pareado solo vale sobre los documentos que vieron TODAS las celdas
    # del factorial. Las referencias (ADR-003) pueden correr sobre un prefijo
    # menor de la misma muestra —LED cuesta minutos por artículo— sin que eso
    # recorte el análisis del factorial: se comparan solo en sus documentos.
    docs = df.groupby("config_id")["doc_id"].apply(set)
    principales = docs.drop(args.referencias, errors="ignore")
    comunes = set.intersection(*(principales if len(principales) else docs))
    fuera = df["config_id"].isin(args.referencias) | df["doc_id"].isin(comunes)
    if (~fuera).any():
        print(
            f"AVISO: solo {len(comunes)} documentos son comunes a todas las "
            "configuraciones del factorial; el análisis se restringe a ellos.\n"
        )
    df = df[
        fuera & (df["doc_id"].isin(comunes) | df["config_id"].isin(args.referencias))
    ]

    vacios = (df["resumen_generado"].str.strip() == "").sum()
    if vacios:
        print(f"AVISO: {vacios} resúmenes vacíos.\n")

    print(
        "Calificando" + (" (ROUGE)" if args.sin_bertscore else " (ROUGE + BERTScore)")
    )
    calificado = calificar(df, con_bertscore=not args.sin_bertscore)
    calificado.drop(columns=["resumen_generado", "resumen_referencia"]).to_csv(
        salida / "calificado.csv", index=False
    )

    resumen = resumir_por_configuracion(calificado)
    resumen.to_csv(salida / "resumen.csv")
    print("\n== Calidad y costo por configuración ==\n")
    print(resumen.to_string())

    if calificado["estrato"].notna().any():
        por_estrato = resumir_por_configuracion(calificado, por=("estrato",))
        por_estrato.to_csv(salida / "por_estrato.csv")
        print("\n== ROUGE-Lsum por estrato de longitud ==\n")
        print(
            por_estrato["rougeLsum"]
            .unstack("estrato")
            .droplevel(["modelo", "estrategia"])
            .to_string()
        )

    metricas = [m for m in ("rougeLsum", "rouge2", "bertscore") if m in calificado]
    pareadas = pruebas_pareadas(calificado, metricas)
    if not pareadas.empty:
        pareadas.to_csv(salida / "pareadas.csv", index=False)
        print("\n== Comparaciones pareadas (diferencia = a − b) ==\n")
        print(
            pareadas[
                ["a", "b", "metrica", "n", "diferencia", "ic95_bajo", "ic95_alto",
                 "gana_a", "p_holm", "significativo"]
            ].to_string(index=False)
        )  # fmt: skip

    if args.baseline in calificado["config_id"].unique():
        # Sobre los documentos del baseline: si LED corrió un prefijo de 60,
        # todas las celdas se comparan contra él en esos mismos 60.
        docs_ref = set(
            calificado.loc[calificado["config_id"] == args.baseline, "doc_id"]
        )
        en_ref = calificado[calificado["doc_id"].isin(docs_ref)]
        recuperacion = recuperacion_frente_a(
            resumir_por_configuracion(en_ref), args.baseline
        )
        recuperacion.to_csv(salida / "recuperacion.csv")
        print(
            f"\n== Frente a {args.baseline}, sobre sus {len(docs_ref)} documentos "
            "(100 % = baseline) ==\n"
        )
        print(recuperacion.to_string())

    print(f"\nTablas en {salida}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
