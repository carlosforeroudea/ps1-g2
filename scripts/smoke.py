"""Smoke test del entorno de desarrollo (semana 5).

Verifica de punta a punta que el entorno está montado y, de paso,
produce la primera medida empírica del proyecto: **la brecha real entre
el tamaño de los artículos y la ventana de contexto de los modelos**.

El anteproyecto cita 4.938 palabras por artículo (Cohan et al., 2018) y
el ADR-002 registra ~5.905 tokens por espacios en el split de test. Ese
número aún subestima el problema, porque los modelos cuentan en tokens
de subpalabra. Esto lo mide con el tokenizer real de BART.

Uso:
    make smoke
    uv run python scripts/smoke.py --n 20
"""

from __future__ import annotations

import argparse
import sys

CHECKPOINT = "facebook/bart-large-cnn"
DATASET = "ccdv/arxiv-summarization"
DATASET_CONFIG = "section"  # ADR-002: preserva los límites de sección
SPLIT = "test"  # ADR-002: único split en uso, no hay fine-tuning


def _paso(titulo: str) -> None:
    print(f"\n\033[1m>> {titulo}\033[0m")


def revisar_dispositivo() -> str:
    import torch

    if torch.cuda.is_available():
        dispositivo = "cuda"
    elif torch.backends.mps.is_available():
        dispositivo = "mps"
    else:
        dispositivo = "cpu"

    print(f"   torch {torch.__version__} · dispositivo disponible: {dispositivo}")
    if dispositivo == "cpu":
        print("   (CPU: suficiente para desarrollo; las corridas van a nube)")
    return dispositivo


def ventana_de_contexto(checkpoint: str) -> int:
    """Ventana de contexto real del modelo, en tokens.

    La fuente correcta es la **config del modelo**, no el tokenizer.
    `tokenizer.model_max_length` devuelve un centinela gigantesco
    (~1e30) cuando el `tokenizer_config.json` del checkpoint no lo
    declara, que es el caso de `facebook/bart-large-cnn`. Confiar en él
    haría que `Truncation` no truncara nunca y que toda la comparación
    del proyecto midiera algo que no ocurre.

    Este es el comportamiento que debe implementar
    `SummarizerModel.context_window` en Fase 3.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)

    for atributo in ("max_position_embeddings", "n_positions"):
        valor = getattr(config, atributo, None)
        if isinstance(valor, int) and 0 < valor < 1_000_000:
            return valor

    raise RuntimeError(
        f"No se pudo determinar la ventana de contexto de {checkpoint}. "
        "Modelos con atención relativa (LongT5) no la declaran y "
        "requieren un límite explícito por configuración."
    )


def revisar_modelo():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    ventana = ventana_de_contexto(CHECKPOINT)
    print(f"   {CHECKPOINT}: ventana de contexto = {ventana:,} tokens")

    centinela = tokenizer.model_max_length
    if not isinstance(centinela, int) or centinela > 1_000_000:
        print(
            "   (nota: tokenizer.model_max_length no es fiable en este "
            f"checkpoint — devuelve {centinela:.3g}. Se usa la config.)"
        )

    if ventana != 1024:
        print(
            f"   \033[33mAVISO\033[0m: se esperaban 1024 tokens y son {ventana}. "
            "Todo el planteamiento del problema asume 1024; "
            "verificar el checkpoint antes de seguir."
        )
    return tokenizer, ventana


def cargar_muestra(n: int):
    """Carga `n` artículos en streaming, sin bajar los 7,26 GB (ADR-002)."""
    from datasets import load_dataset

    try:
        flujo = load_dataset(DATASET, DATASET_CONFIG, split=SPLIT, streaming=True)
    except Exception as exc:
        print(
            f"   \033[33mNo se pudo cargar la config '{DATASET_CONFIG}': {exc}\033[0m"
        )
        print("   Reintentando con la configuración por defecto...")
        flujo = load_dataset(DATASET, split=SPLIT, streaming=True)

    # El corpus NO trae campo `id` (solo `article` y `abstract`), así
    # que el identificador se sintetiza por posición en el split. Es
    # estable mientras el dataset no se reordene, y es lo que permite
    # auditar qué documento produjo cada fila del experimento.
    articulos = []
    for i, ejemplo in enumerate(flujo):
        if i >= n:
            break
        articulos.append({**ejemplo, "doc_id": f"{SPLIT}-{i:05d}"})
    return articulos


def medir_brecha(articulos: list, tokenizer, ventana: int) -> None:
    conteos = [
        len(tokenizer(a["article"], truncation=False)["input_ids"]) for a in articulos
    ]
    conteos.sort()

    n = len(conteos)
    promedio = sum(conteos) / n
    mediana = conteos[n // 2]
    caben = sum(1 for c in conteos if c <= ventana)

    print(f"   Artículos medidos:     {n}")
    print(f"   Tokens (promedio):     {promedio:,.0f}")
    print(f"   Tokens (mediana):      {mediana:,}")
    print(f"   Tokens (mín – máx):    {conteos[0]:,} – {conteos[-1]:,}")
    print(f"   Ventana del modelo:    {ventana:,}")
    print()
    print(f"   \033[1mBrecha: {promedio / ventana:.1f}× la ventana\033[0m")
    print(f"   \033[1mCaben completos: {caben}/{n} ({caben / n:.0%})\033[0m")

    descartado = max(0.0, 1 - ventana / promedio)
    print(f"   \033[1mTruncar descarta ~{descartado:.0%} del artículo promedio\033[0m")
    print()
    print("   Ese porcentaje es la información que el truncamiento pierde,")
    print("   y lo que map-reduce y extractivo-abstractivo deben recuperar")
    print("   para justificar su costo adicional (ADR-003).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=20, help="artículos a medir (por defecto 20)"
    )
    args = parser.parse_args()

    print("\033[1mSmoke test — Resumidor de artículos científicos\033[0m")

    _paso("1/4 Dispositivo de cómputo")
    revisar_dispositivo()

    _paso("2/4 Modelo preentrenado y ventana de contexto")
    tokenizer, ventana = revisar_modelo()

    _paso(f"3/4 Corpus ({DATASET}, split={SPLIT}, streaming)")
    articulos = cargar_muestra(args.n)
    print(f"   {len(articulos)} artículos cargados")
    print(f"   Campos disponibles: {sorted(articulos[0].keys())}")

    _paso("4/4 Brecha de contexto medida con el tokenizer real")
    medir_brecha(articulos, tokenizer, ventana)

    print("\n\033[32m\033[1mEntorno OK.\033[0m\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"\n\033[31m\033[1mSmoke test FALLÓ:\033[0m {type(exc).__name__}: {exc}")
        print(
            "\nRevisa docs/entorno.md. Causas frecuentes: falta `uv sync`, "
            "sin conexión a internet, o el dataset cambió de estructura."
        )
        sys.exit(1)
