"""Esquema y validación de las configuraciones de experimento.

RT-6 (`docs/arquitectura.md`): una configuración inválida debe fallar **antes**
de la corrida, no durante. Un checkpoint mal escrito tiene que reventar en
segundos, no doce horas después.

Cada celda del factorial es un YAML en `experiments/configs/` (ADR-005 §4).
"""

from __future__ import annotations

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Estrategias implementadas. `MapReduce`, `ExtractiveAbstractive` y `LeadK`
# son Fase 3: aparecer aquí es lo que las habilita.
ESTRATEGIAS_IMPLEMENTADAS = frozenset({"truncation"})


class ModeloConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    checkpoint: str
    device: str = "auto"
    num_beams: int = Field(default=4, ge=1, le=16)


class EstrategiaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # map-reduce añade parámetros propios

    kind: str
    max_new_tokens: int = Field(default=256, ge=16, le=1024)

    @field_validator("kind")
    @classmethod
    def _implementada(cls, v: str) -> str:
        if v not in ESTRATEGIAS_IMPLEMENTADAS:
            raise ValueError(
                f"Estrategia {v!r} todavía no implementada. "
                f"Disponibles: {sorted(ESTRATEGIAS_IMPLEMENTADAS)}."
            )
        return v


class MuestraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    config: str | None = None
    split: str = "test"
    n: int = Field(ge=1)
    seed: int


class ExperimentoConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # `metrics` y anotaciones futuras

    id: str
    model: ModeloConfig
    strategy: EstrategiaConfig
    sample: MuestraConfig


def cargar_config(ruta: str | pathlib.Path) -> ExperimentoConfig:
    """Lee y valida un YAML de experimento."""
    ruta = pathlib.Path(ruta)
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ValueError(f"{ruta} no contiene un mapeo YAML válido.")
    return ExperimentoConfig(**datos)
