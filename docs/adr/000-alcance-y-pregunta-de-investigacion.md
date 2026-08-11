# ADR-000 — Alcance y pregunta de investigación
Estado: Aceptado | Fecha: 2026-08-11 | Autores: Forero, Martelo, Mercado

## Contexto
El banco de ideas plantea un "resumidor de artículos científicos" con
modelos preentrenados y sin fine-tuning. Los modelos abstractivos
disponibles tienen ventanas de 512–1024 tokens; un artículo de arXiv
promedia >5.000 tokens. Esta brecha es el problema técnico central.

## Decisión
Encuadrar el proyecto como investigación aplicada. Pregunta:
¿qué estrategia de manejo de contexto largo maximiza la calidad del
resumen bajo modelos preentrenados con ventana limitada, y cuál es el
compromiso calidad/costo operativo?
Variable independiente: estrategia (truncamiento | map-reduce |
extractivo-abstractivo) × modelo (BART | PEGASUS | LongT5/LED).
Variable dependiente: ROUGE-1/2/Lsum, BERTScore, latencia p50/p95,
pico de memoria, ratio de compresión.

## Fuera de alcance
Fine-tuning. Entrenamiento desde cero. Idiomas distintos del inglés.
Resumen multi-documento. Despliegue en nube productiva.

## Consecuencias
(+) Resultado válido aunque el ROUGE sea bajo: el hallazgo es el trade-off.
(+) Costo de cómputo acotado: solo inferencia.
(−) Exige rigor estadístico (n≥300, tests pareados), no basta una demo.
