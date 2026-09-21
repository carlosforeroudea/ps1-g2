# Resumidor inteligente de artículos científicos

**Grupo 2 — Los Predictores** · Proyecto Integrador I (2508700) · UdeA 2026-2

Los artículos científicos de arXiv promedian **7.839 tokens** medidos con
el tokenizer de BART. BART y PEGASUS aceptan **1.024**: la brecha es de
**7,7×**, y truncar descarta ~87 % del artículo.

Este proyecto estudia qué estrategia de manejo de contexto largo
—truncamiento, map-reduce o extractivo-abstractivo— ofrece el mejor
compromiso entre calidad del resumen y costo computacional usando
modelos preentrenados de contexto corto, y despliega el resultado como
una plataforma web de acceso público.

La configuración que la plataforma sirve en producción no se elige por
conveniencia: es la que gane el criterio experimental de la Fase 3.

Integrantes: Naneth Martelo · Carlos Forero · Elio Mercado
Asesor: Antonio Jesús Tamayo Herrera

## Empezar

Requiere [uv](https://docs.astral.sh/uv/). Si no lo tienes:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
make setup    # monta el entorno exacto desde uv.lock
make test     # tests (no usan red)
make smoke    # verifica el entorno y mide la brecha de contexto real
```

`make smoke` descarga el tokenizer de BART y unos pocos artículos del
corpus en streaming, y reporta cuántos tokens tienen de verdad y qué
fracción cabe en la ventana del modelo. Si corre, el entorno está bien.

Guía detallada y solución de problemas: [docs/entorno.md](docs/entorno.md).

## Estructura

```
src/resumidor/     Canalización: dominio, modelos, estrategias, evaluación
experiments/       Configuraciones del factorial y resultados crudos
scripts/           Utilidades (smoke test)
tests/             Tests unitarios
docs/adr/          Decisiones de arquitectura
docs/arquitectura.md
paper/             Informe final
```

## Documentación

| Documento | Contenido |
|---|---|
| [docs/arquitectura.md](docs/arquitectura.md) | Diseño del sistema, requisitos, protocolos |
| [ADR-000](docs/adr/000-alcance-y-pregunta-de-investigacion.md) | Alcance y pregunta de investigación |
| [ADR-001](docs/adr/001-despliegue-en-alcance.md) | El despliegue entra en alcance |
| [ADR-002](docs/adr/002-corpus-experimental.md) | Corpus experimental |
| [ADR-003](docs/adr/003-rol-de-led-y-longt5.md) | Rol de LED y LongT5 |
| [ADR-004](docs/adr/004-stack-y-entorno.md) | Stack técnico y entorno |
| [ADR-005](docs/adr/005-arquitectura-de-la-canalizacion.md) | Arquitectura de la canalización |

## Estado

Fase 2 (semanas 5–6), *Diseño y preparación*.

Hecho: arquitectura y ADRs, entorno reproducible, EDA del corpus completo,
y la infraestructura de ejecución — runner dirigido por YAML, adaptador de
Hugging Face (BART y PEGASUS), instrumentación de costo y registro de
resultados por documento.

```bash
make prueba    # corrida local de 3 documentos con BART
uv run python -u scripts/run_experiment.py experiments/configs/truncation_pegasus.yaml --limite 3
```

En curso: muestra experimental estratificada de 300 artículos.
Sigue: estrategias map-reduce y extractivo-abstractivo, y el factorial
completo (Fase 3, semanas 7–14).
