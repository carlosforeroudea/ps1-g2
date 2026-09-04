# ADR-001 — El despliegue en producción entra en alcance
Estado: Aceptado | Fecha: 2026-09-03 | Autores: Forero, Martelo, Mercado
Supersede parcialmente: ADR-000 (sección "Fuera de alcance")

## Contexto
El ADR-000 (2026-08-11) declaró "Despliegue en nube productiva" fuera
de alcance, bajo el encuadre de investigación aplicada donde el
entregable era la evidencia del trade-off.

El anteproyecto avalado dice lo contrario en dos lugares:
- Resumen: "Este proyecto desarrolla un resumidor de artículos
  científicos desplegado en producción y de acceso público".
- Objetivo específico 8: "Desarrollar una plataforma web que permita
  al usuario cargar artículos científicos, seleccionar opciones
  predefinidas de procesamiento y obtener los resúmenes generados".
- Tipo de proyecto: **Desarrollo** (no investigación pura).

El cronograma reserva la Fase 4 (semanas 12–16) a "Plataforma,
validación y cierre", con producto "Producto final funcional,
integrado en la plataforma web".

El anteproyecto es el documento posterior y avalado por el profesor.
Prevalece.

## Decisión
El despliegue operativo entra en alcance como entregable de Fase 4.

Consecuencias sobre el ADR-000: se anula la línea "Despliegue en nube
productiva" de su sección "Fuera de alcance". Todo lo demás del
ADR-000 sigue vigente (fine-tuning, entrenamiento desde cero, idiomas
distintos del inglés y resumen multi-documento continúan fuera).

El encuadre del proyecto es mixto y ambas mitades son obligatorias:
1. Evidencia experimental (objetivos 1–7) — determina la configuración.
2. Producto desplegado (objetivos 8–9) — la sirve públicamente.

La configuración que la plataforma ejecuta en producción no se elige
por popularidad ni por conveniencia de implementación: es la que gane
el criterio de compromiso calidad/costo de la Fase 3. Ese es el
vínculo que justifica que ambas mitades sean el mismo proyecto.

## Consecuencias
(+) El objetivo específico 8 deja de estar en conflicto con el ADR-000.
(+) Fija un requisito arquitectónico duro: la plataforma web debe
    consumir la misma canalización que los experimentos, no una
    reimplementación. Ver ADR-005.
(−) Añade requisitos no funcionales que la investigación pura no
    tenía: latencia aceptable para un usuario interactivo, límite de
    tamaño de carga, y costo de inferencia sostenido en el tiempo.
(−) El presupuesto de cómputo en nube ($1.000.000) debe repartirse
    entre las corridas experimentales y el servicio desplegado.
