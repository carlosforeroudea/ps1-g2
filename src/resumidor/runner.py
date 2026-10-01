"""Runner de experimentos: ejecuta una configuración y registra el resultado.

Recibe toda la configuración desde el YAML (ADR-005 §4): añadir una celda del
factorial es un archivo nuevo, no código nuevo.

**Escribe una fila por documento, en cuanto termina cada uno.** No acumula en
memoria para volcar al final. Dos razones, ambas del diseño adoptado:

1. En GPU *spot*, una expropiación cuesta un documento, no la corrida entera.
2. Los tests pareados del ADR-000 necesitan el resultado por documento, no
   solo el agregado.
"""

from __future__ import annotations

import itertools
import json
import pathlib
from collections.abc import Iterator
from datetime import UTC, datetime

from resumidor.config import ExperimentoConfig
from resumidor.data.corpus import cargar_muestra
from resumidor.domain import CostMetrics, Document, SummaryResult
from resumidor.instrumentation import medir, metodo_memoria
from resumidor.models.counting import ContadorDeInvocaciones
from resumidor.models.huggingface import HFSummarizer
from resumidor.strategies import ESTRATEGIAS
from resumidor.strategies.base import ContextStrategy


def construir_estrategia(config: ExperimentoConfig) -> ContextStrategy:
    parametros = config.strategy.model_dump(exclude={"kind"})
    clase = ESTRATEGIAS[config.strategy.kind]
    admitidos = {c for c in clase.__dataclass_fields__}
    # Los desconocidos ya los rechazó la validación (RT-6); aquí solo se
    # descarta `max_new_tokens` en estrategias que no generan (`LeadK`).
    return clase(**{k: v for k, v in parametros.items() if k in admitidos})


def construir_modelo(config: ExperimentoConfig) -> HFSummarizer:
    return HFSummarizer(
        config.model.checkpoint,
        dispositivo=config.model.device,
        num_beams=config.model.num_beams,
        ventana=config.model.ventana,
        # `exclude_none`: solo se envían los parámetros declarados, para no
        # imponer valores por defecto nuestros sobre los del checkpoint
        # cuando la configuración no dice nada.
        generacion=config.model.generation.model_dump(exclude_none=True),
    )


def ejecutar_documento(
    documento: Document,
    modelo: HFSummarizer,
    estrategia: ContextStrategy,
) -> SummaryResult:
    """Resume un documento midiendo el costo de la estrategia completa.

    La instrumentación envuelve `estrategia.summarize`, no `modelo.generate`
    (ADR-005 §3): medir el modelo haría que map-reduce reportara el costo de
    una invocación de N.
    """
    tokens_entrada = modelo.count_tokens(documento.text)
    contador = ContadorDeInvocaciones(modelo)

    with medir(modelo.dispositivo) as medicion:
        resumen = estrategia.summarize(documento, contador)

    return SummaryResult(
        summary=resumen,
        cost=CostMetrics(
            latency_seconds=medicion.latencia_s,
            peak_memory_mb=medicion.pico_memoria_mb,
            model_calls=contador.invocaciones,
            input_tokens=tokens_entrada,
            output_tokens=modelo.count_tokens(resumen),
        ),
        model_name=modelo.name,
        strategy_name=estrategia.name,
    )


def _fila(
    config: ExperimentoConfig,
    documento: Document,
    resultado: SummaryResult,
    dispositivo: str,
    metodo: str,
) -> dict:
    """Una fila de resultado, autocontenida y auditable.

    Lleva la configuración que la produjo para que el archivo se pueda leer
    sin el YAML al lado.
    """
    return {
        "config_id": config.id,
        "doc_id": documento.doc_id,
        "estrato": documento.estrato,
        "modelo": resultado.model_name,
        "estrategia": resultado.strategy_name,
        # El dispositivo RESUELTO, no el configurado: con `device: auto` la
        # fila diría "auto" y la latencia no sería comparable entre corridas.
        "dispositivo": dispositivo,
        "resumen_generado": resultado.summary,
        "resumen_referencia": documento.reference_summary,
        "latencia_s": round(resultado.cost.latency_seconds, 3),
        "pico_memoria_mb": round(resultado.cost.peak_memory_mb, 1),
        "metodo_memoria": metodo,
        "invocaciones_modelo": resultado.cost.model_calls,
        "tokens_entrada": resultado.cost.input_tokens,
        "tokens_salida": resultado.cost.output_tokens,
        "ratio_compresion": round(resultado.compression_ratio, 5),
        "semilla": config.sample.seed,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _hechos(config: ExperimentoConfig, salida: pathlib.Path) -> set[str]:
    """Documentos ya escritos en `salida` por esta misma configuración.

    Si el archivo tiene filas de otra configuración o de otra semilla, falla:
    mezclarlas en el mismo archivo rompería el pareado sin avisar.
    """
    if not salida.exists():
        return set()
    hechos: set[str] = set()
    for linea in salida.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        fila = json.loads(linea)
        if (fila["config_id"], fila["semilla"]) != (config.id, config.sample.seed):
            raise RuntimeError(
                f"{salida} contiene filas de {fila['config_id']!r} "
                f"(semilla {fila['semilla']}), no de {config.id!r} "
                f"(semilla {config.sample.seed}). Usa otra salida o --desde-cero."
            )
        hechos.add(fila["doc_id"])
    return hechos


def ejecutar(
    config: ExperimentoConfig,
    salida: pathlib.Path,
    *,
    limite: int | None = None,
    streaming: bool = False,
    reanudar: bool = True,
) -> Iterator[dict]:
    """Ejecuta la configuración y va escribiendo `salida` documento a documento.

    `limite` acota la muestra sin tocar el YAML: sirve para la prueba local
    pequeña sin inventar una configuración paralela que luego diverja. Se
    aplica como **prefijo** de la muestra completa, así que los documentos de
    una prueba con `--limite 3` son los mismos tres con que empieza la
    corrida real.

    Con `reanudar` (por defecto) salta los documentos que ya están en
    `salida` y añade el resto. Es lo que hace que una expropiación de la GPU
    *spot* cueste un documento y no la corrida: basta relanzar el mismo job.
    """
    hechos = _hechos(config, salida) if reanudar else set()

    documentos = cargar_muestra(
        dataset=config.sample.dataset,
        config=config.sample.config,
        split=config.sample.split,
        n=config.sample.n,
        seed=config.sample.seed,
        streaming=streaming,
        muestreo=config.sample.muestreo,
        estratos=config.sample.estratos,
    )
    pendientes = [
        d
        for d in itertools.islice(documentos, limite or config.sample.n)
        if d.doc_id not in hechos
    ]
    if not pendientes:
        return

    modelo = construir_modelo(config)
    estrategia = construir_estrategia(config)

    # Antes de medir nada: descarga, carga y calentamiento del grafo. Sin
    # el calentamiento, el primer documento absorbe el costo de compilación
    # del dispositivo (448,8 s frente a 28,3 s medidos en la semana 6).
    modelo.precargar(pesos=getattr(estrategia, "usa_modelo_generativo", True))

    medida_memoria = metodo_memoria(modelo.dispositivo)

    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("a" if reanudar else "w", encoding="utf-8") as f:
        for documento in pendientes:
            resultado = ejecutar_documento(documento, modelo, estrategia)
            fila = _fila(
                config, documento, resultado, modelo.dispositivo, medida_memoria
            )
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()  # el punto de la escritura incremental
            yield fila
