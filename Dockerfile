# Imagen del runner de experimentos (Fase 3).
#
# Es la pieza que hace auditable la Fase 3: el digest de esta imagen, junto
# con uv.lock y la semilla de la configuración, fija el entorno exacto que
# produjo cada resultado. Es lo que un cuaderno de Colab no puede entregar.
#
# Construir:
#   docker build -t resumidor:dev .
# Probar en local (2 documentos, CPU):
#   docker run --rm -v "$PWD/experiments/results:/app/experiments/results" \
#       resumidor:dev experiments/configs/truncation_bart.yaml --limite 2

FROM python:3.11-slim

# uv fijado por versión: si la imagen se reconstruye en seis meses, debe
# resolver el mismo árbol de dependencias.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    # La caché de modelos dentro de la imagen; en Vertex AI el contenedor es
    # efímero y cada job vuelve a descargar los pesos desde el Hub.
    HF_HOME=/app/.cache/huggingface

# Dependencias primero, en su propia capa: el código cambia mucho más a
# menudo que el lock, y así la reconstrucción no repite la descarga de torch
# (que con las bibliotecas de CUDA son varios GB).
#
# README.md va aquí porque `pyproject.toml` lo declara en `readme`: sin él,
# hatchling falla al construir el paquete con
# `OSError: Readme file does not exist`.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ src/
COPY scripts/ scripts/
COPY experiments/configs/ experiments/configs/
RUN uv sync --locked --no-dev

# Verificación en tiempo de construcción: importa todos los módulos y valida
# las configuraciones. Sin esto, un archivo excluido por .dockerignore no se
# nota hasta que el job de Vertex AI falla minutos después, y el error
# aparece en un log remoto en vez de aquí.
RUN /app/.venv/bin/python -c "\
import resumidor.runner, resumidor.eval, resumidor.data.corpus; \
from resumidor.config import cargar_config; \
import pathlib; \
cfgs = sorted(pathlib.Path('experiments/configs').glob('*.yaml')); \
assert cfgs, 'no se copiaron las configuraciones'; \
[cargar_config(c) for c in cfgs]; \
print(f'OK: modulos importables y {len(cfgs)} configuraciones validas')"

# Se llama al intérprete del entorno directamente, sin pasar por `uv run`:
# `uv run` re-sincroniza en cada arranque y descarga las dependencias de
# desarrollo (jupyterlab, ruff, debugpy) dentro del job, que no las necesita.
ENTRYPOINT ["/app/.venv/bin/python", "-u", "scripts/run_experiment.py"]
CMD ["experiments/configs/truncation_bart.yaml", "--limite", "2"]
