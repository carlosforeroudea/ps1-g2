"""RT-6: una configuración inválida falla antes de la corrida, no durante."""

from __future__ import annotations

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from resumidor.config import cargar_config

BASE = {
    "id": "prueba",
    "model": {"kind": "bart", "checkpoint": "facebook/bart-large-cnn"},
    "strategy": {"kind": "truncation", "max_new_tokens": 400},
    "sample": {
        "dataset": "ccdv/arxiv-summarization",
        "split": "test",
        "n": 300,
        "seed": 20262,
    },
}


def _escribir(tmp_path: pathlib.Path, datos: dict) -> pathlib.Path:
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos), encoding="utf-8")
    return ruta


def _variante(**cambios) -> dict:
    datos = {k: (v.copy() if isinstance(v, dict) else v) for k, v in BASE.items()}
    for clave, valor in cambios.items():
        seccion, campo = clave.split("__")
        datos[seccion][campo] = valor
    return datos


def test_configuracion_valida(tmp_path: pathlib.Path) -> None:
    c = cargar_config(_escribir(tmp_path, BASE))
    assert c.id == "prueba"
    assert c.model.checkpoint == "facebook/bart-large-cnn"
    assert c.model.device == "auto"  # valor por defecto
    assert c.strategy.max_new_tokens == 400


def test_las_configuraciones_del_repo_son_validas() -> None:
    configs = sorted(pathlib.Path("experiments/configs").glob("*.yaml"))
    assert configs, "no hay configuraciones en experiments/configs/"
    for ruta in configs:
        assert cargar_config(ruta).id == ruta.stem


def test_rechaza_estrategia_no_implementada(tmp_path: pathlib.Path) -> None:
    """map_reduce está diseñada pero no implementada: debe fallar ya.

    Sin esto, una corrida de 300 documentos arrancaría y moriría tarde.
    """
    ruta = _escribir(tmp_path, _variante(strategy__kind="map_reduce"))
    with pytest.raises(ValidationError, match="todavía no implementada"):
        cargar_config(ruta)


def test_rechaza_campo_desconocido_en_el_modelo(tmp_path: pathlib.Path) -> None:
    # Un typo como `checkpint:` no debe pasar silenciosamente.
    ruta = _escribir(tmp_path, _variante(model__checkpint="x"))
    with pytest.raises(ValidationError):
        cargar_config(ruta)


@pytest.mark.parametrize("valor", [0, -1, 2000])
def test_rechaza_max_new_tokens_fuera_de_rango(
    tmp_path: pathlib.Path, valor: int
) -> None:
    ruta = _escribir(tmp_path, _variante(strategy__max_new_tokens=valor))
    with pytest.raises(ValidationError):
        cargar_config(ruta)


def test_rechaza_muestra_vacia(tmp_path: pathlib.Path) -> None:
    ruta = _escribir(tmp_path, _variante(sample__n=0))
    with pytest.raises(ValidationError):
        cargar_config(ruta)
