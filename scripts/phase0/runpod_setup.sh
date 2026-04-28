#!/usr/bin/env bash
set -euo pipefail

# Phase 0 RunPod GPU provisioning and experiment execution.
# Requires: runpodctl CLI, pass password manager, rsync
#
# Usage:
#   ./scripts/phase0/runpod_setup.sh [provision|run|collect|teardown|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/phase0"

RUNPOD_API_KEY="$(pass runpod/api_key)" || { echo "ERROR: Failed to retrieve RunPod API key from pass" >&2; exit 1; }
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: RunPod API key is empty" >&2; exit 1
fi
export RUNPOD_API_KEY

GPU_TYPE="NVIDIA A100 80GB PCIe"
DOCKER_IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
VOLUME_SIZE=50
DISK_SIZE=20

POD_NAMES=("tract-phase0-bge" "tract-phase0-gte" "tract-phase0-deberta")
POD_IDS_FILE="$SCRIPT_DIR/.pod_ids"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

validate_pod_id() {
    local pod_id="$1"
    if [[ ! "$pod_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: Invalid pod_id: $pod_id" >&2; exit 1
    fi
}

provision() {
    log "Provisioning 3 GPU pods..."
    mkdir -p "$RESULTS_DIR"
    > "$POD_IDS_FILE"

    for name in "${POD_NAMES[@]}"; do
        log "Creating pod: $name"
        pod_id=$(runpodctl create pod \
            --name "$name" \
            --gpuType "$GPU_TYPE" \
            --gpuCount 1 \
            --imageName "$DOCKER_IMAGE" \
            --volumeSize "$VOLUME_SIZE" \
            --containerDiskSize "$DISK_SIZE" \
            --ports "22/tcp" \
            2>&1 | grep -oP 'pod "\K[^"]+')
        echo "$name=$pod_id" >> "$POD_IDS_FILE"
        log "Created $name: $pod_id"
    done

    log "Waiting for pods to be ready..."
    sleep 30

    while IFS='=' read -r name pod_id; do
        log "Waiting for $name ($pod_id)..."
        for i in $(seq 1 30); do
            status=$(runpodctl get pod "$pod_id" 2>/dev/null | grep -oP 'status: \K\w+' || echo "unknown")
            if [ "$status" = "RUNNING" ]; then
                log "$name is RUNNING"
                break
            fi
            sleep 10
        done
    done < "$POD_IDS_FILE"
}

setup_pod() {
    local pod_id="$1"
    local name="$2"
    validate_pod_id "$pod_id"

    log "Setting up $name ($pod_id)..."

    local -a ssh_cmd=(runpodctl ssh --podId "$pod_id")

    "${ssh_cmd[@]}" "pip install -q sentence-transformers transformers torch numpy scipy scikit-learn"

    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/scripts/phase0/" /workspace/scripts/phase0/
    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/tract/" /workspace/tract/
    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/data/" /workspace/data/
    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/pyproject.toml" /workspace/pyproject.toml

    "${ssh_cmd[@]}" "cd /workspace && pip install -e '.[phase0]'"
}

run_experiments() {
    log "Setting up pods and running experiments..."

    local bge_id gte_id deberta_id
    while IFS='=' read -r name pod_id; do
        case "$name" in
            tract-phase0-bge) bge_id="$pod_id" ;;
            tract-phase0-gte) gte_id="$pod_id" ;;
            tract-phase0-deberta) deberta_id="$pod_id" ;;
        esac
    done < "$POD_IDS_FILE"

    while IFS='=' read -r name pod_id; do
        setup_pod "$pod_id" "$name" &
    done < "$POD_IDS_FILE"
    wait

    log "Phase A: Running baseline experiments in parallel..."
    runpodctl ssh --podId "$bge_id" "cd /workspace && python -m scripts.phase0.exp1_embedding_baseline --model bge" &
    runpodctl ssh --podId "$gte_id" "cd /workspace && python -m scripts.phase0.exp1_embedding_baseline --model gte" &
    runpodctl ssh --podId "$deberta_id" "cd /workspace && python -m scripts.phase0.exp1_embedding_baseline --model deberta" &
    wait
    log "Phase A complete."

    log "Phase B: Running path-enriched experiments..."
    runpodctl ssh --podId "$bge_id" "cd /workspace && python -m scripts.phase0.exp3_hierarchy_paths --model bge" &
    runpodctl ssh --podId "$gte_id" "cd /workspace && python -m scripts.phase0.exp3_hierarchy_paths --model gte" &
    wait
    log "Phase B complete."

    log "Phase C: Running description pilot on best model..."
    runpodctl ssh --podId "$bge_id" "cd /workspace && python -m scripts.phase0.exp4_hub_descriptions --model all"
    log "Phase C complete."
}

collect() {
    log "Collecting results from pods..."
    mkdir -p "$RESULTS_DIR"

    while IFS='=' read -r name pod_id; do
        log "Collecting from $name ($pod_id)..."
        runpodctl recv --podId "$pod_id" /workspace/results/phase0/ "$RESULTS_DIR/" || true
    done < "$POD_IDS_FILE"

    log "Results collected to $RESULTS_DIR"
    ls -la "$RESULTS_DIR"
}

teardown() {
    log "Tearing down pods..."

    while IFS='=' read -r name pod_id; do
        log "Removing $name ($pod_id)..."
        runpodctl remove pod "$pod_id" || true
    done < "$POD_IDS_FILE"

    rm -f "$POD_IDS_FILE"
    log "All pods removed."
}

case "${1:-all}" in
    provision) provision ;;
    run) run_experiments ;;
    collect) collect ;;
    teardown) teardown ;;
    all)
        provision
        run_experiments
        collect
        teardown
        ;;
    *) echo "Usage: $0 [provision|run|collect|teardown|all]"; exit 1 ;;
esac
