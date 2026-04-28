"""RunPod API provisioning: create pod, poll for SSH, terminate.

Direct REST + GraphQL API calls — no runpodctl CLI dependency.
Adapted from proven pattern in ai-security-framework-crosswalk.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Final

import requests

logger = logging.getLogger(__name__)

GRAPHQL_URL: Final[str] = "https://api.runpod.io/graphql"
REST_URL: Final[str] = "https://rest.runpod.io/v1"
SSH_POLL_TIMEOUT_S: Final[int] = 600
SSH_POLL_INTERVAL_S: Final[int] = 15

GPU_PREFERENCE: Final[list[str]] = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H100 PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
]


def _get_api_key() -> str:
    result = subprocess.run(
        ["pass", "runpod/api-key"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    key = result.stdout.strip()
    if not key:
        raise ValueError("RunPod API key is empty")
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _gql(query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(GRAPHQL_URL, headers=_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data and data["errors"]:
        critical = [e for e in data["errors"] if "lowestPrice" not in str(e.get("path", []))]
        if critical:
            raise RuntimeError(f"GraphQL errors: {critical}")
    return data.get("data", {})


def list_available_gpus(min_vram_gb: int = 48) -> list[dict]:
    data = _gql("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud } }")
    return [
        g for g in data.get("gpuTypes", [])
        if (g.get("memoryInGb") or 0) >= min_vram_gb
        and (g.get("communityCloud") or g.get("secureCloud"))
    ]


def find_fastest_available(min_vram_gb: int = 48) -> str:
    gpus = list_available_gpus(min_vram_gb)
    available_ids = {g["id"] for g in gpus}

    for pref in GPU_PREFERENCE:
        if pref in available_ids:
            return pref

    if gpus:
        gpus.sort(key=lambda g: -(g.get("memoryInGb") or 0))
        return gpus[0]["id"]

    raise RuntimeError(f"No GPU with >= {min_vram_gb}GB VRAM available")


def _validate_pod_id(pod_id: str) -> None:
    if not re.match(r'^[a-zA-Z0-9_-]+$', pod_id):
        raise ValueError(f"Invalid pod_id: {pod_id}")


def create_pod(
    gpu_type_id: str,
    name: str,
    image: str = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
    gpu_count: int = 1,
    volume_gb: int = 50,
    container_disk_gb: int = 20,
) -> dict:
    """Create a RunPod pod. Returns {pod_id, ip, port, gpu_type, name}."""
    for cloud_type in ["SECURE", "COMMUNITY"]:
        payload = {
            "name": name,
            "imageName": image,
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": gpu_count,
            "cloudType": cloud_type,
            "volumeInGb": volume_gb,
            "containerDiskInGb": container_disk_gb,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
        }
        resp = requests.post(
            f"{REST_URL}/pods",
            headers=_headers(), json=payload, timeout=30,
        )
        data = resp.json()
        if isinstance(data, dict) and data.get("id"):
            pod = data
            break
        err = data[0]["error"] if isinstance(data, list) else data.get("error", "")
        logger.warning("%s cloud: %s", cloud_type, err)
    else:
        raise RuntimeError(f"Failed to create pod on any cloud type: {data}")

    pod_id = pod["id"]
    _validate_pod_id(pod_id)
    logger.info("Pod created: %s (%s) — waiting for SSH...", pod_id, gpu_type_id)

    ssh_info = _wait_for_ssh(pod_id)

    return {
        "pod_id": pod_id,
        "ip": ssh_info["ip"],
        "port": ssh_info["port"],
        "gpu_type": gpu_type_id,
        "name": name,
    }


def _wait_for_ssh(pod_id: str) -> dict:
    _validate_pod_id(pod_id)
    start = time.time()
    while time.time() - start < SSH_POLL_TIMEOUT_S:
        resp = requests.get(
            f"{REST_URL}/pods/{pod_id}",
            headers=_headers(), timeout=30,
        )
        resp.raise_for_status()
        pod = resp.json()
        ip = pod.get("publicIp", "")
        port_mappings = pod.get("portMappings", {})
        ssh_port = port_mappings.get("22")
        if ip and ssh_port:
            try:
                s = socket.create_connection((ip, int(ssh_port)), timeout=5)
                s.close()
                return {"ip": ip, "port": int(ssh_port)}
            except (OSError, socket.timeout):
                pass
        elapsed = int(time.time() - start)
        status = pod.get("desiredStatus", "unknown")
        logger.info(
            "[%ds] Pod %s: %s, ip=%s, ssh_port=%s",
            elapsed, pod_id, status, ip or "pending", ssh_port or "pending",
        )
        time.sleep(SSH_POLL_INTERVAL_S)

    raise TimeoutError(f"Pod {pod_id} SSH not ready within {SSH_POLL_TIMEOUT_S}s")


def get_running_pods() -> list[dict]:
    data = _gql(
        "query { myself { pods { id name desiredStatus "
        "runtime { ports { ip publicPort privatePort } } "
        "machine { gpuDisplayName } } } }"
    )
    myself = data.get("myself", {})
    return [p for p in myself.get("pods", []) if p.get("desiredStatus") == "RUNNING"]


def terminate_pod(pod_id: str) -> None:
    _validate_pod_id(pod_id)
    _gql(
        "mutation terminatePod($input: PodTerminateInput!) { podTerminate(input: $input) }",
        {"input": {"podId": pod_id}},
    )
    logger.info("Terminated pod %s", pod_id)


def terminate_all() -> None:
    for pod in get_running_pods():
        terminate_pod(pod["id"])


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="RunPod provisioner")
    parser.add_argument("--list", action="store_true", help="List available GPUs")
    parser.add_argument("--fastest", action="store_true", help="Find fastest available GPU")
    parser.add_argument("--create", type=str, help="Create pod with GPU type ID")
    parser.add_argument("--running", action="store_true", help="List running pods")
    parser.add_argument("--terminate", type=str, help="Terminate pod by ID")
    parser.add_argument("--terminate-all", action="store_true", help="Terminate all pods")
    args = parser.parse_args()

    if args.list:
        for g in list_available_gpus():
            cc = "comm" if g.get("communityCloud") else ""
            sc = "secure" if g.get("secureCloud") else ""
            print(f"  {g['id']:<36} {g.get('memoryInGb', '?'):>4}GB  {cc} {sc}")
    elif args.fastest:
        print(f"Fastest available: {find_fastest_available()}")
    elif args.create:
        result = create_pod(args.create, name="tract-phase0-test")
        print(json.dumps(result, indent=2))
    elif args.running:
        for pod in get_running_pods():
            ports = pod.get("runtime", {}).get("ports", [])
            ssh_port = next((p for p in ports if p.get("privatePort") == 22), {})
            ip = ssh_port.get("ip", "pending")
            port = ssh_port.get("publicPort", "?")
            gpu = pod.get("machine", {}).get("gpuDisplayName", "?")
            print(f"  {pod['id']}: {pod.get('name', '?')} — {gpu} @ {ip}:{port}")
    elif args.terminate:
        terminate_pod(args.terminate)
    elif args.terminate_all:
        terminate_all()
