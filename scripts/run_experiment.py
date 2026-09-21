"""Ejecuta una configuración de experimento.

Uso:
    # prueba local pequeña (tarjeta "Infraestructura y ejecución de modelos")
    uv run python scripts/run_experiment.py \
        experiments/configs/truncation_bart.yaml --limite 3

    # celda completa
    uv run python scripts/run_experiment.py experiments/configs/truncation_bart.yaml

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
    p.add_argument("config", type=pathlib.Path, help="YAML de la celda")
    p.add_argument(
        "--limite", type=int, default=None, help="procesa como mucho N documentos"
    )
    p.add_argument(
        "--salida",
        type=pathlib.Path,
        default=None,
        help="archivo .jsonl (por defecto experiments/results/<id>.jsonl)",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="no descarga la partición; reproducible pero no uniforme",
    )
    args = p.parse_args()

    # Validación primero: un checkpoint mal escrito debe reventar ahora,
    # no después de descargar 4 GB de pesos (RT-6).
    config = cargar_config(args.config)
    salida = args.salida or pathlib.Path("experiments/results") / f"{config.id}.jsonl"

    print(f"\033[1mConfiguración:\033[0m {config.id}")
    print(f"  modelo     {config.model.checkpoint}")
    print(
        f"  estrategia {config.strategy.kind}"
        f" · max_new_tokens={config.strategy.max_new_tokens}"
    )
    print(
        f"  muestra    {config.sample.split} · n={args.limite or config.sample.n}"
        f" · semilla={config.sample.seed}"
    )
    print(f"  salida     {salida}")
    print(
        "\nCargando modelo y corpus (la primera vez descarga; luego va de caché)...\n"
    )

    total_latencia = 0.0
    n = 0
    for fila in ejecutar(config, salida, limite=args.limite, streaming=args.streaming):
        n += 1
        total_latencia += fila["latencia_s"]
        print(
            f"\033[1m[{n}] {fila['doc_id']}\033[0m  "
            f"{fila['tokens_entrada']:,} → {fila['tokens_salida']} tokens  ·  "
            f"{fila['latencia_s']:.1f} s  ·  "
            f"{fila['invocaciones_modelo']} invocación(es)"
        )
        print(f"    {fila['resumen_generado'][:180]}...")
        print()

    if n == 0:
        print("\033[31mNo se procesó ningún documento.\033[0m")
        return 1

    print(
        f"\033[32m\033[1m{n} documentos · {total_latencia:.1f} s "
        f"({total_latencia / n:.1f} s/documento)\033[0m"
    )
    print(f"Resultados en {salida}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
