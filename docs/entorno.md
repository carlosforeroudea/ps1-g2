# Entorno de desarrollo

Guía de montaje para el equipo. Las decisiones detrás de este stack
están en [ADR-004](adr/004-stack-y-entorno.md).

## 1. Instalar uv

`uv` gestiona tanto la versión de Python como las dependencias. No hace
falta instalar Python 3.11 aparte: `uv` lo descarga solo.

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cierra y reabre la terminal, y comprueba: `uv --version`.

## 2. Montar el entorno

```bash
git clone https://github.com/carlosforeroudea/ps1-g2.git
cd ps1-g2
make setup
```

`make setup` ejecuta `uv sync`, que lee `uv.lock` y reconstruye el árbol
de dependencias **exacto** —mismas versiones para los tres integrantes,
para CI y para la nube—. Crea `.venv/` en la raíz; no hace falta
activarlo, `uv run` lo usa automáticamente.

La primera vez descarga PyTorch (varios cientos de MB) y tarda unos
minutos. Después es cuestión de segundos.

## 3. Comprobar que funciona

```bash
make test     # tests unitarios, sin red
make smoke    # descarga tokenizer + corpus, mide la brecha de contexto
```

`make smoke` es la verificación real: si termina en verde, el entorno
está completo. Imprime cuántos tokens tienen los artículos según el
tokenizer de BART y qué fracción del documento descarta el
truncamiento.

## 4. Hooks de pre-commit (recomendado)

```bash
make hooks
```

Corre lint y formato antes de cada commit, así CI no falla por detalles
de estilo.

## Comandos frecuentes

| Comando | Qué hace |
|---|---|
| `make setup` | Reconstruye el entorno desde el lock |
| `make test` | Tests (sin red) |
| `make lint` | Verifica lint y formato |
| `make format` | Aplica el formato |
| `make smoke` | Verifica el entorno de punta a punta |
| `make clean` | Borra cachés de herramientas |

Para correr algo puntual sin activar el venv: `uv run python ...`,
`uv run pytest ...`.

## Añadir una dependencia

```bash
uv add nombre-del-paquete          # runtime
uv add --dev nombre-del-paquete    # solo desarrollo
```

Esto actualiza `pyproject.toml` **y** `uv.lock`. **Ambos se commitean.**
CI corre `uv sync --locked` y falla si divergen — es deliberado: impide
que alguien introduzca una dependencia que los demás no tienen.

> **Advertencia del ADR-004.** Actualizar `transformers`, `rouge-score`
> o `bert-score` a mitad de la Fase 3 puede mover las métricas e
> invalidar las comparaciones ya corridas. Si hay que hacerlo, se
> repiten **todas** las celdas del factorial, no solo las nuevas.

## Problemas frecuentes

**`uv: command not found`** — La terminal no recargó el PATH. Ábrela de
nuevo, o `source $HOME/.local/bin/env`.

**`make smoke` tarda ~1 minuto en el paso 3/4** — Es normal y está
verificado: `datasets` resuelve el parquet remoto del split de test
antes de entregar el primer artículo. No descarga el corpus completo.

**Aviso `You are sending unauthenticated requests to the HF Hub`** —
Inofensivo. El corpus y los checkpoints son públicos. Solo afecta a los
límites de tasa; si molesta, define `HF_TOKEN` con un token de lectura.

**`make smoke` falla al cargar el dataset** — Suele ser falta de
conexión, o que `ccdv/arxiv-summarization` cambió de estructura. El
script reintenta con la configuración por defecto e informa qué pasó.

**La primera corrida de `make smoke` tarda mucho** — Descarga los pesos
del tokenizer y unos artículos. Quedan en la caché de Hugging Face
(`~/.cache/huggingface`); la segunda vez es inmediata.

**El corpus ocupa demasiado** — No debería: el smoke test usa
`streaming=True` y no baja los 7,26 GB. Si ves descargas grandes,
alguien quitó el streaming (ver [ADR-002](adr/002-corpus-experimental.md)).

**Todo está lento en mi máquina** — Es esperado: en local se trabaja con
muestras pequeñas en CPU/MPS. Las corridas experimentales completas van
a la nube (ADR-004).
