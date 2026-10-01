"""Carga de la muestra experimental desde el Hub (ADR-002).

Sobre la configuración `section` del corpus: **no delimita secciones**, pese
a su nombre. Verificado sobre el propio dataset —`document` entrega el
artículo como una sola línea; `section`, como ~220 líneas que son *oraciones*,
sin títulos de sección—. Sigue siendo la configuración correcta, pero por otra
razón: da fronteras de oración limpias, que es lo que necesitan la selección
extractiva y la fragmentación de map-reduce para no cortar a mitad de frase.

Mientras `Section` siga modelando secciones que el corpus no tiene, este
cargador entrega el artículo como una sola sección. Ver la nota en
`docs/adr/002-corpus-experimental.md`.
"""

from __future__ import annotations

from collections.abc import Iterator

from resumidor.domain import Document, Section


def _a_documento(
    ejemplo: dict, indice: int, split: str, estrato: str | None = None
) -> Document:
    """Convierte una fila del corpus en un `Document`.

    El corpus no trae campo identificador (ADR-002): el `doc_id` se sintetiza
    por posición, que es lo que permite auditar qué documento produjo cada
    fila del experimento.
    """
    return Document(
        doc_id=f"{split}-{indice:05d}",
        sections=(Section(title="", text=ejemplo["article"]),),
        reference_summary=ejemplo.get("abstract"),
        estrato=estrato,
    )


def indices_estratificados(
    longitudes: list[int], *, n: int, estratos: int, seed: int
) -> list[tuple[int, str]]:
    """Muestreo estratificado por cuantiles de longitud: `(índice, estrato)`.

    Es el criterio del notebook 03 —cuartiles de la longitud en palabras y el
    mismo número de artículos por estrato—, con una diferencia deliberada: se
    aplica sobre el split de **test**, no sobre train. `pegasus-arxiv` y
    `led-large-16384-arxiv` están afinados sobre el train de este corpus;
    evaluarlos sobre artículos que vieron al entrenar inflaría su ROUGE y
    sesgaría toda comparación contra BART.

    El orden de salida se baraja con la misma semilla: así cualquier prefijo
    de la muestra (`--limite`, o una corrida interrumpida) sigue cubriendo
    todos los estratos en lugar de solo los artículos cortos.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    longitudes_arr = np.asarray(longitudes)
    cortes = np.quantile(longitudes_arr, np.linspace(0, 1, estratos + 1))
    # `searchsorted` sobre los cortes interiores asigna 0..estratos-1.
    asignacion = np.searchsorted(cortes[1:-1], longitudes_arr, side="right")

    por_estrato = n // estratos
    elegidos: list[tuple[int, str]] = []
    for e in range(estratos):
        miembros = np.flatnonzero(asignacion == e)
        if len(miembros) < por_estrato:
            raise ValueError(
                f"El estrato Q{e + 1} tiene {len(miembros)} artículos; "
                f"se piden {por_estrato}."
            )
        tomados = rng.choice(miembros, size=por_estrato, replace=False)
        elegidos.extend((int(i), f"Q{e + 1}") for i in sorted(tomados))

    orden = rng.permutation(len(elegidos))
    return [elegidos[i] for i in orden]


def cargar_muestra(
    *,
    dataset: str,
    config: str | None,
    split: str,
    n: int,
    seed: int,
    streaming: bool = False,
    muestreo: str = "aleatorio",
    estratos: int = 4,
) -> Iterator[Document]:
    """Devuelve `n` documentos reproducibles del corpus.

    Con `streaming=False` (por defecto) descarga **solo los archivos de la
    partición pedida** y hace un muestreo aleatorio real con `seed`.

    El `data_files` explícito no es un adorno: `load_dataset(repo, split="test")`
    descarga *todas* las particiones del config y luego selecciona, lo que en
    este corpus son 7,26 GB para usar 6.440 artículos. Verificado en la semana
    6 —se bajaron 3,4 GB de `train` antes de detenerlo—. Apuntar a
    `<config>/<split>-*.parquet` baja solo lo que hace falta.

    Con `streaming=True` no descarga nada, pero el barajado opera sobre un
    búfer y no sobre la partición entera: la muestra es reproducible pero no
    uniforme. Útil para una prueba rápida, no para el experimento definitivo.

    Con `muestreo="estratificado"` aplica `indices_estratificados` y el
    `doc_id` es el índice real del artículo en el split (ADR-002), no su
    posición dentro de la muestra.
    """
    from datasets import load_dataset

    patron = f"{config}/{split}-*.parquet" if config else f"{split}-*.parquet"
    fuente = load_dataset(
        dataset,
        data_files={split: patron},
        split=split,
        streaming=streaming,
        # Los metadatos del corpus declaran tres particiones; al pedir solo
        # una, la verificación de particiones falla. Desactivarla es la
        # consecuencia de bajar únicamente `test`, no un atajo: las sumas de
        # verificación de los archivos que sí se descargan siguen aplicándose.
        verification_mode="no_checks",
    )

    if muestreo == "estratificado":
        if streaming:
            raise ValueError(
                "El muestreo estratificado necesita las longitudes de toda la "
                "partición; no es compatible con streaming."
            )
        longitudes = [len(a.split()) for a in fuente["article"]]
        for indice, estrato in indices_estratificados(
            longitudes, n=n, estratos=estratos, seed=seed
        ):
            yield _a_documento(fuente[indice], indice, split, estrato)
        return

    if streaming:
        barajado = fuente.shuffle(seed=seed, buffer_size=max(n * 10, 100))
        for i, ejemplo in enumerate(barajado.take(n)):
            yield _a_documento(ejemplo, i, split)
        return

    indices = fuente.shuffle(seed=seed).select(range(min(n, len(fuente))))
    for i, ejemplo in enumerate(indices):
        yield _a_documento(ejemplo, i, split)
