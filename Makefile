# Resumidor de artículos científicos — Grupo 2, Los Predictores
# Requiere uv. Instalación y primeros pasos: docs/entorno.md

.PHONY: help setup test lint format smoke prueba factorial metricas evaluar estado estado-seguir hooks clean

help:
	@echo "Targets disponibles:"
	@echo "  setup    Monta el entorno exacto desde uv.lock"
	@echo "  test     Corre los tests (sin red)"
	@echo "  lint     Lint y verificación de formato"
	@echo "  format   Aplica el formato"
	@echo "  smoke    Verifica el entorno y mide la brecha de contexto real"
	@echo "  prueba   Corrida local de 3 documentos con BART"
	@echo "  factorial  Las 9 configuraciones sobre la muestra de 300 (reanudable)"
	@echo "  metricas   ROUGE, BERTScore, pruebas pareadas -> experiments/results/analisis/"
	@echo "  evaluar  Ejecuta el notebook de evaluación con los resultados"
	@echo "  estado   Avance de experimentos, build y jobs de GCP"
	@echo "  estado-seguir  Lo mismo, refrescando cada 20 s"
	@echo "  hooks    Instala los hooks de pre-commit"
	@echo "  clean    Borra cachés de herramientas"

setup:
	uv sync

test:
	uv run pytest -m "not network"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

smoke:
	uv run python -u scripts/smoke.py

# Prueba local pequeña: 3 documentos con BART. `-u` desactiva el búfer de
# stdout para ver el avance en vivo y no solo al terminar.
prueba:
	uv run python -u scripts/run_experiment.py \
		experiments/configs/truncation_bart.yaml --limite 3

# Todas las configuraciones, de la más barata a la más cara. Reanuda: si se
# interrumpe, relanzar continúa donde quedó cada celda. En un portátil son
# decenas de horas en un M4. `caffeinate -i` evita que el Mac se suspenda.
CONFIGS_FACTORIAL = lead_k truncation_bart extractive_abstractive_bart \
	truncation_pegasus extractive_abstractive_pegasus map_reduce_bart \
	map_reduce_pegasus control_longt5_sin_afinar baseline_led

factorial:
	PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i uv run python -u scripts/run_experiment.py \
		$(addprefix experiments/configs/,$(addsuffix .yaml,$(CONFIGS_FACTORIAL)))

metricas:
	uv run python scripts/evaluar.py

# Ejecuta el notebook de evaluación de punta a punta y deja las salidas
# embebidas. Requiere que existan resultados en experiments/results/.
evaluar:
	uv run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/evaluacion_modelos.ipynb

# Estado de un vistazo: experimentos, build e infraestructura en GCP.
estado:
	@bash scripts/estado.sh

# Lo mismo, refrescando cada 20 s. Ctrl-C para salir.
estado-seguir:
	@while true; do clear; bash scripts/estado.sh; sleep 20; done

hooks:
	uv run pre-commit install

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
