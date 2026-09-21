# Resumidor de artículos científicos — Grupo 2, Los Predictores
# Requiere uv. Instalación y primeros pasos: docs/entorno.md

.PHONY: help setup test lint format smoke prueba hooks clean

help:
	@echo "Targets disponibles:"
	@echo "  setup    Monta el entorno exacto desde uv.lock"
	@echo "  test     Corre los tests (sin red)"
	@echo "  lint     Lint y verificación de formato"
	@echo "  format   Aplica el formato"
	@echo "  smoke    Verifica el entorno y mide la brecha de contexto real"
	@echo "  prueba   Corrida local de 3 documentos con BART"
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

hooks:
	uv run pre-commit install

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
