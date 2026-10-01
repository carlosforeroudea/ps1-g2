#!/usr/bin/env bash
# Lanza un Custom Job de Vertex AI (L4 spot) por cada configuración dada.
#
#   bash scripts/lanzar_gcp.sh map_reduce_bart map_reduce_pegasus
#
# Las configuraciones van dentro de la imagen: reconstrúyela antes si
# cambiaste algo en experiments/configs/ (docs/gcp.md, «Deuda de diseño»).
# Relanzar una configuración reanuda su .jsonl en el bucket.

set -euo pipefail

PROYECTO="${PROYECTO:-udea-509713}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-udea-509713-resumidor}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ $# -gt 0 ]] || { echo "uso: $0 CONFIG [CONFIG...]" >&2; exit 2; }

for cfg in "$@"; do
    [[ -f "$REPO/experiments/configs/$cfg.yaml" ]] \
        || { echo "no existe experiments/configs/$cfg.yaml" >&2; exit 1; }
    job="$(mktemp -t "job-$cfg")"
    sed -e "s/PROYECTO/$PROYECTO/g" -e "s/REGION/$REGION/g" \
        -e "s/BUCKET/$BUCKET/g" -e "s/CONFIG/$cfg/g" \
        "$REPO/deploy/gcp/job-gpu-l4.yaml" > "$job"
    gcloud ai custom-jobs create --region="$REGION" \
        --display-name="resumidor-${cfg//_/-}" --config="$job"
done
