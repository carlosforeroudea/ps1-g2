"""Ejecuta una configuración de experimento.

Uso:
    # prueba local pequeña (tarjeta "Infraestructura y ejecución de modelos")
    uv run python scripts/run_experiment.py \
        experiments/configs/truncation_bart.yaml --limite 3

    # celda completa (reanuda si ya hay filas en la salida)
    uv run python scripts/run_experiment.py experiments/configs/truncation_bart.yaml

    # varias celdas en secuencia: todo el factorial y las referencias
    uv run python scripts/run_experiment.py experiments/configs/*.yaml

La configuración sale entera del YAML (ADR-005 §4). `--limite` acota la muestra
sin tocar el archivo, para no crear una configuración paralela que luego
diverja de la real.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from resumidor.config import cargar_config
from resumidor.runner import ejecutar


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "configs", type=pathlib.Path, nargs="+", help="YAML de la(s) celda(s)"
    )
    p.add_argument(
        "--limite", type=int, default=None, help="procesa como mucho N documentos"
    )
    p.add_argument(
        "--salida",
        type=pathlib.Path,
        default=None,
        help="archivo .jsonl (por defecto experiments/results/<id>.jsonl); "
        "solo con una configuración",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="no descarga la partición; reproducible pero no uniforme",
    )
    p.add_argument(
        "--desde-cero",
        action="store_true",
        help="sobrescribe la salida en vez de reanudar donde quedó",
    )
    args = p.parse_args()

    if args.salida and len(args.configs) > 1:
        p.error("--salida solo tiene sentido con una configuración")

    # Validación primero, de TODAS: un checkpoint mal escrito en la última
    # celda debe reventar ahora, no tras horas de las anteriores (RT-6).
    configs = [cargar_config(ruta) for ruta in args.configs]

    fallidas = 0
    for config in configs:
        salida = (
            args.salida or pathlib.Path("experiments/results") / f"{config.id}.jsonl"
        )
        if correr(config, salida, args) != 0:
            fallidas += 1
    return 1 if fallidas else 0


def correr(config, salida: pathlib.Path, args: argparse.Namespace) -> int:
    print(f"\033[1mConfiguración:\033[0m {config.id}")
    print(f"  modelo     {config.model.checkpoint}")
    print(
        f"  estrategia {config.strategy.kind}"
        f" · max_new_tokens={config.strategy.max_new_tokens}"
    )
    print(
        f"  muestra    {config.sample.split} · n={args.limite or config.sample.n}"
        f" · {config.sample.muestreo} · semilla={config.sample.seed}"
    )
    print(f"  salida     {salida}")
    print(
        "\nCargando modelo y corpus (la primera vez descarga; luego va de caché)...\n"
    )

    total_latencia = 0.0
    n = 0
    for fila in ejecutar(
        config,
        salida,
        limite=args.limite,
        streaming=args.streaming,
        reanudar=not args.desde_cero,
    ):
        n += 1
        total_latencia += fila["latencia_s"]
        print(
            f"\033[1m[{n}] {fila['doc_id']}\033[0m ({fila['estrato'] or '-'})  "
            f"{fila['tokens_entrada']:,} → {fila['tokens_salida']} tokens  ·  "
            f"{fila['latencia_s']:.1f} s  ·  "
            f"{fila['invocaciones_modelo']} invocación(es)"
        )
        print(f"    {fila['resumen_generado'][:180]}...")
        print()

    if n == 0:
        if salida.exists():
            print(f"\033[32mNada pendiente: {salida} ya está completo.\033[0m\n")
            return 0
        print("\033[31mNo se procesó ningún documento.\033[0m")
        return 1

    print(
        f"\033[32m\033[1m{n} documentos · {total_latencia:.1f} s "
        f"({total_latencia / n:.1f} s/documento)\033[0m"
    )
    print(f"Resultados en {salida}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
