# ADR-004 — Stack técnico y entorno de desarrollo
Estado: Aceptado | Fecha: 2026-09-03 | Autores: Forero, Martelo, Mercado

## Contexto
El ADR-000 exige rigor estadístico y resultados reproducibles. Un
experimento cuyas dependencias no están fijadas no es reproducible:
un cambio de versión menor en `transformers` o en `rouge-score` puede
mover las métricas lo suficiente como para invalidar una comparación
pareada.

Son tres integrantes con máquinas distintas, y las corridas
experimentales pesadas irán a la nube. El entorno tiene que ser
idéntico en los cuatro sitios.

## Decisión
**Lenguaje:** Python 3.11. No 3.12 ni 3.13: el ecosistema de PyTorch y
las dependencias de métricas tienen soporte más maduro y ruedas
precompiladas disponibles en 3.11, y no necesitamos nada de 3.12.

**Gestor de dependencias:** `uv`, con `pyproject.toml` y `uv.lock`
ambos versionados en Git.

Frente a conda: `uv` produce un lock determinista y multiplataforma
que fija el árbol completo de dependencias transitivas. Un
`environment.yml` de conda sin lock explícito resuelve distinto en
distintas fechas y máquinas, que es exactamente el modo de fallo que
el ADR-000 no puede permitirse. `uv` además instala en segundos, lo
que importa cuando tres personas montan el entorno y CI lo monta en
cada push.

**Núcleo:** PyTorch, `transformers` (modelos preentrenados),
`datasets` (corpus del ADR-002).

**Métricas de calidad:** `rouge-score` (ROUGE-1/2/Lsum) y `bert-score`
(BERTScore), vía `evaluate`. Son las implementaciones de referencia
que usa la literatura citada; reimplementar ROUGE es una fuente
clásica de cifras no comparables.

**Análisis:** `numpy`, `pandas`, `scipy` (tests pareados del ADR-000),
`matplotlib`.

**Calidad de código:** `ruff` (lint y formato en una sola herramienta),
`pytest`, `pre-commit`.

**Ejecución:** desarrollo local en CPU/MPS con muestras pequeñas;
corridas experimentales completas y baseline LED/LongT5 en nube
(ADR-003). El código no debe asumir CUDA: selección de dispositivo en
un solo lugar, con CPU como opción siempre válida.

**Layout:** `src/` (paquete `resumidor`), que evita que los tests
importen accidentalmente el árbol de trabajo en vez del paquete
instalado.

## Consecuencias
(+) `uv sync` reconstruye el entorno exacto desde el lock, en local y
    en nube. La reproducibilidad exigida por el ADR-000 queda
    mecanizada, no confiada a la disciplina del equipo.
(+) Una sola herramienta (`ruff`) para lint y formato reduce la
    fricción de configuración en un equipo pequeño.
(−) `uv` es menos conocido que conda en entornos académicos; requiere
    un paso de instalación documentado (ver `docs/entorno.md`).
(−) Fijar versiones significa que actualizarlas es una decisión
    deliberada. Actualizar `transformers` o `rouge-score` a mitad de
    la Fase 3 invalida las comparaciones ya corridas: si hay que
    hacerlo, se repiten **todas** las celdas.
(−) Desarrollo en CPU/MPS es lento para inferencia real. Mitigación:
    muestras pequeñas en local, corridas completas en nube.
