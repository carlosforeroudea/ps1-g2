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

Fase 3, *Experimentación*.

Implementadas las cuatro estrategias del ADR-005 —`Truncation`, `MapReduce`,
`ExtractiveAbstractive` y `LeadK`— y las 9 configuraciones del diseño: el
factorial 3 estrategias × {BART, PEGASUS}, el techo LED, el control LongT5 y
el piso `lead_k`. Todas sobre la misma muestra estratificada de 300 artículos
del split de test (75 por cuartil de longitud, ADR-002).

```bash
make prueba      # 3 documentos con BART, para verificar el entorno
make factorial   # las 9 configuraciones × 300 documentos (reanudable)
make metricas    # ROUGE, BERTScore, pruebas pareadas → experiments/results/analisis/
```

Para ejecutarlo en local (Mac con Apple Silicon, sin GPU dedicada: usa MPS),
[notebooks/04_ejecucion_factorial.ipynb](notebooks/04_ejecucion_factorial.ipynb)
corre todo, muestra el avance con tiempo restante estimado y genera las
métricas al final. En un M4 el factorial completo son del orden de 30–40 horas
(map-reduce hace ~9 invocaciones por artículo); se puede repartir en varias
sesiones. Necesita ~8 GB libres para los modelos.

Alternativa en la nube: `bash scripts/lanzar_gcp.sh map_reduce_bart ...`, un
job L4 spot por configuración ([docs/gcp.md](docs/gcp.md)).

Las corridas son reanudables: relanzar una configuración continúa donde quedó.
