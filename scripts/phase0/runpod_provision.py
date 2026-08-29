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
from typing import Any, Final

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

# RunPod's two tiers. SECURE is RunPod's own datacentres; COMMUNITY is
# third-party hosts, and cheaper for that reason.
CLOUD_TYPE_SECURE: Final[str] = "SECURE"
CLOUD_TYPE_COMMUNITY: Final[str] = "COMMUNITY"
# create_pod asks for these in order and takes the first that has capacity.
# This tuple is AUTHORITATIVE: it is what the fleet actually runs on, and
# get_gpu_price prices its head rather than asking a separate question.
CLOUD_TYPE_PREFERENCE: Final[tuple[str, ...]] = (
    CLOUD_TYPE_SECURE, CLOUD_TYPE_COMMUNITY,
)
# The tier every budget number is computed against. Pricing the preferred tier
# rather than the cross-cloud lowest matters because the two diverge: measured
# live on 2026-08-26, an H100 80GB was $2.69/hr unfiltered and $3.29/hr with
# secureCloud:true, so the unfiltered figure understated the fleet by 22.3%.
# Pricing the head of the preference order is also the conservative choice --
# SECURE is the dearer tier, so a fallback to COMMUNITY can only come in under
# the budget that was already checked.
PRICE_CLOUD_TYPE: Final[str] = CLOUD_TYPE_PREFERENCE[0]


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


def _gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(GRAPHQL_URL, headers=_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data and data["errors"]:
        critical = [e for e in data["errors"] if "lowestPrice" not in str(e.get("path", []))]
        if critical:
            raise RuntimeError(f"GraphQL errors: {critical}")
    result: dict[str, Any] = data.get("data", {})
    return result


def list_available_gpus(min_vram_gb: int = 48) -> list[dict[str, Any]]:
    data = _gql("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud } }")
    return [
        g for g in data.get("gpuTypes", [])
        if (g.get("memoryInGb") or 0) >= min_vram_gb
        and (g.get("communityCloud") or g.get("secureCloud"))
    ]


def get_gpu_price(gpu_type_id: str, gpu_count: int = 1) -> float:
    """On-demand USD/hour for one pod of this GPU type, on the tier we buy.

    Raises if the price cannot be read. A run whose hourly rate is unknown
    cannot be checked against a budget, and "unknown" must not silently become
    "free" -- that is how a fleet of the most expensive part on the market gets
    provisioned against a ceiling that exists only in prose.

    The query is filtered to PRICE_CLOUD_TYPE rather than asking for the
    cross-cloud lowest price. Without the filter this priced a pod the code
    does not prefer: create_pod asks for SECURE first, and on 2026-08-26 the
    unfiltered lowest for an H100 80GB was $2.69/hr against $3.29/hr secure --
    a 22.3% understatement carried into every budget check, every "worst case"
    line in the log, and the campaign plan built on them.
    """
    data = _gql(
        "query GpuPrice($id: String, $input: GpuLowestPriceInput) { "
        "  gpuTypes(input: {id: $id}) { "
        "    id displayName "
        "    lowestPrice(input: $input) { uninterruptablePrice minimumBidPrice } "
        "  } "
        "}",
        {
            "id": gpu_type_id,
            "input": {
                "gpuCount": gpu_count,
                "secureCloud": PRICE_CLOUD_TYPE == CLOUD_TYPE_SECURE,
            },
        },
    )
    types = data.get("gpuTypes") or []
    if not types:
        raise RuntimeError(f"No price information returned for GPU type {gpu_type_id!r}")

    lowest = types[0].get("lowestPrice") or {}
    price = lowest.get("uninterruptablePrice")
    if price is None:
        raise RuntimeError(
            f"RunPod returned no on-demand price for {gpu_type_id!r} "
            f"(lowestPrice={lowest!r}). Refusing to provision against an "
            "unknown hourly rate."
        )
    return float(price)


def find_fastest_available(
    min_vram_gb: int = 48,
    max_usd_per_hour: float | None = None,
    gpu_count: int = 1,
) -> str:
    """Select a GPU type, honouring the preference order and a price ceiling.

    Args:
        max_usd_per_hour: Reject any GPU type above this on-demand rate. The
            previous fallback was "largest VRAM wins", which can select a part
            several times the rate of an H100 when the preferred types are all
            unavailable.
    """
    gpus = list_available_gpus(min_vram_gb)
    if not gpus:
        raise RuntimeError(f"No GPU with >= {min_vram_gb}GB VRAM available")

    available_ids = {g["id"] for g in gpus}
    ordered = [p for p in GPU_PREFERENCE if p in available_ids]
    # Preference order first, then remaining candidates by VRAM.
    ordered += [
        g["id"] for g in sorted(gpus, key=lambda g: -(g.get("memoryInGb") or 0))
        if g["id"] not in GPU_PREFERENCE
    ]

    rejected: list[str] = []
    for gpu_id in ordered:
        try:
            price = get_gpu_price(gpu_id, gpu_count)
        except RuntimeError as exc:
            logger.warning("Skipping %s: %s", gpu_id, exc)
            rejected.append(f"{gpu_id} (no price)")
            continue
        if max_usd_per_hour is not None and price > max_usd_per_hour:
            logger.warning("Skipping %s: $%.2f/hr exceeds the $%.2f/hr ceiling",
                           gpu_id, price, max_usd_per_hour)
            rejected.append(f"{gpu_id} (${price:.2f}/hr)")
            continue
        logger.info("Selected GPU %s at $%.2f/hr", gpu_id, price)
        return gpu_id

    raise RuntimeError(
        f"No GPU with >= {min_vram_gb}GB VRAM is available within "
        f"${max_usd_per_hour}/hr. Rejected: {rejected}"
    )


# RunPod's error text when a type is listed but has no free instances.
CAPACITY_ERROR_MARKERS: Final[tuple[str, ...]] = (
    "no instances currently available",
    "no longer any instances available",
)


def is_capacity_error(exc: BaseException) -> bool:
    """Whether a pod-creation failure means "try a different GPU type".

    list_available_gpus reports the types that EXIST, not the types with free
    capacity, so a perfectly reasonable selection can fail at create time.
    That is a transient supply condition rather than a bug, and it should cost
    a different GPU rather than the whole campaign.
    """
    text = str(exc).lower()
    return any(marker in text for marker in CAPACITY_ERROR_MARKERS)


def rank_available_gpus(
    min_vram_gb: int = 48,
    max_usd_per_hour: float | None = None,
    gpu_count: int = 1,
) -> list[tuple[str, float]]:
    """Every acceptable GPU type as (id, usd_per_hour), best first.

    Same ordering and same price ceiling as find_fastest_available, which
    returns only the head of this list. Callers that create pods want the
    tail as well, so a capacity failure on the first choice can fall through
    to the second instead of ending the run.
    """
    gpus = list_available_gpus(min_vram_gb)
    if not gpus:
        raise RuntimeError(f"No GPU with >= {min_vram_gb}GB VRAM available")

    available_ids = {g["id"] for g in gpus}
    ordered = [p for p in GPU_PREFERENCE if p in available_ids]
    ordered += [
        g["id"] for g in sorted(gpus, key=lambda g: -(g.get("memoryInGb") or 0))
        if g["id"] not in GPU_PREFERENCE
    ]

    ranked: list[tuple[str, float]] = []
    for gpu_id in ordered:
        try:
            price = get_gpu_price(gpu_id, gpu_count)
        except RuntimeError:
            continue
        if max_usd_per_hour is not None and price > max_usd_per_hour:
            continue
        ranked.append((gpu_id, price))
    if not ranked:
        raise RuntimeError(
            f"No GPU with >= {min_vram_gb}GB VRAM is available within "
            f"${max_usd_per_hour}/hr."
        )
    return ranked


def _validate_pod_id(pod_id: str) -> None:
    if not re.match(r'^[a-zA-Z0-9_-]+$', pod_id):
        raise ValueError(f"Invalid pod_id: {pod_id}")


def validate_ssh_endpoint(ip: str, port: int) -> tuple[str, int]:
    """Validate an API-supplied SSH endpoint before it reaches a shell.

    publicIp and portMappings come from a remote API and are interpolated into
    shell commands. Parse the address rather than pattern-matching it, so a
    value carrying shell metacharacters cannot reach `ssh`/`rsync` at all.
    Returns the normalised (ip, port).
    """
    import ipaddress

    try:
        address = ipaddress.ip_address(ip)
    except ValueError as exc:
        raise ValueError(
            f"RunPod returned a public IP that is not an IP address: {ip!r}"
        ) from exc

    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f"RunPod returned an out-of-range SSH port: {port!r}")

    return str(address), int(port)


def create_pod(
    gpu_type_id: str,
    name: str,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_count: int = 1,
    volume_gb: int = 50,
    container_disk_gb: int = 20,
) -> dict[str, Any]:
    """Create a RunPod pod with retry.

    Returns {pod_id, ip, port, gpu_type, name, cloud_type}.

    `cloud_type` is which tier of CLOUD_TYPE_PREFERENCE actually accepted the
    pod. It used to go unrecorded, so a fleet that fell back from SECURE to
    COMMUNITY looked identical in the logs and in the state file to one that
    did not. That is not bookkeeping: _rsync_to ships the working tree,
    data/processed/licensed included, to whichever host answered, so the tier
    is a statement about where licensed corpus went. It also explains a price
    that came in under the budget check, which prices SECURE.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        pod = None
        # Captured beside `pod` rather than read back off the response: the tier
        # we ASKED for is the fact worth recording, and it is known here without
        # depending on a field the REST payload may or may not carry.
        landed_cloud = ""
        for cloud_type in CLOUD_TYPE_PREFERENCE:
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
                landed_cloud = cloud_type
                break
            err = data[0]["error"] if isinstance(data, list) else data.get("error", "")
            logger.warning("%s cloud (attempt %d): %s", cloud_type, attempt, err)
        if pod:
            break
        if attempt < max_attempts:
            wait = 5 * attempt
            logger.info("Retrying pod creation in %ds...", wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to create pod after {max_attempts} attempts: {data}")

    pod_id = pod["id"]
    _validate_pod_id(pod_id)
    if landed_cloud != PRICE_CLOUD_TYPE:
        # Loud, because this is the silent half of the fallback: the pod is
        # cheaper than the budget assumed and is running on a host RunPod does
        # not own, and neither fact appeared anywhere before.
        logger.warning(
            "Pod %s (%s) landed on the %s tier, not the preferred %s. The "
            "working tree, licensed corpus included, will be rsynced to it.",
            pod_id, name, landed_cloud, PRICE_CLOUD_TYPE,
        )
    logger.info("Pod created: %s (%s, %s cloud) — waiting for SSH...",
                pod_id, gpu_type_id, landed_cloud)

    try:
        ssh_info = _wait_for_ssh(pod_id)
    except Exception:
        # The pod exists and is billing from the moment it was created. If SSH
        # never comes up there is nothing on it worth keeping and nobody
        # holding its id, so leaving it running bills indefinitely for a host
        # the caller cannot even reach. Terminate before re-raising; a failure
        # to terminate is logged and does not mask the original error.
        logger.error("SSH never came up for pod %s; terminating it.", pod_id)
        try:
            terminate_pod(pod_id)
        except Exception:
            logger.exception(
                "Could not terminate unreachable pod %s. It is STILL BILLING; "
                "terminate it by hand.", pod_id,
            )
        raise

    return {
        "pod_id": pod_id,
        "ip": ssh_info["ip"],
        "port": ssh_info["port"],
        "gpu_type": gpu_type_id,
        "name": name,
        "cloud_type": landed_cloud,
    }


def _wait_for_ssh(pod_id: str) -> dict[str, Any]:
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
            # Validate before the endpoint is stored, so nothing downstream can
            # interpolate an unvalidated API value into a shell command.
            safe_ip, safe_port = validate_ssh_endpoint(ip, int(ssh_port))
            try:
                s = socket.create_connection((safe_ip, safe_port), timeout=5)
                s.close()
                return {"ip": safe_ip, "port": safe_port}
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


def get_running_pods() -> list[dict[str, Any]]:
    data = _gql(
        "query { myself { pods { id name desiredStatus "
        "runtime { ports { ip publicPort privatePort } } "
        "machine { gpuDisplayName } } } }"
    )
    myself = data.get("myself", {})
    return [p for p in myself.get("pods", []) if p.get("desiredStatus") == "RUNNING"]


def create_pods_parallel(
    configs: list[dict[str, str]],
    gpu_type_id: str,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_count: int = 1,
    volume_gb: int = 50,
    container_disk_gb: int = 20,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Create multiple RunPod pods concurrently.

    Each config dict must have 'name' and 'role' keys.
    Returns list of pod info dicts in same order as configs.
    """
    import concurrent.futures

    def _create_one(cfg: dict[str, str]) -> dict[str, Any]:
        pod = create_pod(
            gpu_type_id, name=cfg["name"], image=image,
            gpu_count=gpu_count, volume_gb=volume_gb,
            container_disk_gb=container_disk_gb,
        )
        pod["role"] = cfg["role"]
        # The one line that ties a fold to the host it will train on, so the
        # tier belongs on it rather than only in create_pod's own log.
        logger.info("Ready: %s @ %s:%d (%s cloud) for fold %s",
                    cfg["name"], pod["ip"], pod["port"],
                    pod.get("cloud_type") or "unrecorded", cfg["role"])
        return pod

    workers = min(max_workers, len(configs))
    logger.info("Creating %d pods in parallel (workers=%d)...", len(configs), workers)
    pods: list[dict[str, Any]] = [{}] * len(configs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {
            ex.submit(_create_one, cfg): i
            for i, cfg in enumerate(configs)
        }
        # future.result() used to be called directly here, so the first
        # exception propagated out of the `with` block while the other futures
        # were still creating pods. Those pods were created, billed, and
        # recorded nowhere: the caller never received a list and the state file
        # still said "provisioning" with zero pods. A real run hit this when
        # RunPod ran out of H100 capacity mid-fleet and left three pods
        # orphaned. Collect every outcome first, then decide.
        errors: dict[int, BaseException] = {}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                pods[idx] = future.result()
            except BaseException as exc:  # noqa: BLE001 - collect all, decide after
                errors[idx] = exc
                logger.error("Pod %s failed to create: %s", configs[idx]["name"], exc)

    if errors:
        # A partial fleet cannot produce a LOFO result, and every pod that DID
        # come up is billing for nothing. Give the ones that succeeded back
        # before raising, rather than leaving them for an orphan sweep that
        # only runs if someone is watching.
        created = [p for p in pods if p.get("pod_id")]
        if created:
            logger.warning(
                "Terminating %d pod(s) that came up before the failure.",
                len(created),
            )
            survivors = terminate_pods([p["pod_id"] for p in created])
            if survivors:
                logger.error(
                    "Could not terminate %s. They are STILL BILLING; run "
                    "`runpod_parallel reap --confirm`.", survivors,
                )
        first = sorted(errors)[0]
        raise RuntimeError(
            f"{len(errors)} of {len(configs)} pods failed to create; the "
            f"{len(created)} that succeeded have been terminated. First "
            f"failure ({configs[first]['name']}): {errors[first]}"
        ) from errors[first]

    logger.info("All %d pods created and SSH-ready.", len(pods))
    return pods


def terminate_pod(pod_id: str) -> None:
    _validate_pod_id(pod_id)
    _gql(
        "mutation terminatePod($input: PodTerminateInput!) { podTerminate(input: $input) }",
        {"input": {"podId": pod_id}},
    )
    logger.info("Terminated pod %s", pod_id)


def terminate_pods(pod_ids: list[str]) -> list[str]:
    """Terminate exactly these pods. Returns the ids that failed to terminate.

    Scoped teardown. Prefer this to terminate_all in anything automated: this
    account may be running work that has nothing to do with the caller.
    Continues past a failure so one unreachable pod cannot strand the rest,
    and reports what is still up rather than leaving it billing silently.
    """
    failed: list[str] = []
    for pod_id in pod_ids:
        try:
            terminate_pod(pod_id)
        except Exception as exc:  # noqa: BLE001 - report every failure, stop for none
            logger.error("Failed to terminate pod %s: %s", pod_id, exc)
            failed.append(pod_id)
    if failed:
        logger.error(
            "STILL RUNNING and billing: %s. Terminate them from the RunPod "
            "console or with --terminate <id>.", failed,
        )
    return failed


def terminate_all() -> None:
    """Terminate every running pod on the account.

    Account-wide and indiscriminate: it will kill pods belonging to unrelated
    work. Kept for manual console-style use. Automated teardown should call
    terminate_pods with the ids it created.
    """
    pods = get_running_pods()
    logger.warning("terminate_all: terminating ALL %d running pods on this "
                   "account, including any not created by this run", len(pods))
    terminate_pods([p["id"] for p in pods])


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
            ssh_port: dict[str, Any] = next(
                (p for p in ports if p.get("privatePort") == 22), {}
            )
            ip = ssh_port.get("ip", "pending")
            port = ssh_port.get("publicPort", "?")
            gpu = pod.get("machine", {}).get("gpuDisplayName", "?")
            print(f"  {pod['id']}: {pod.get('name', '?')} — {gpu} @ {ip}:{port}")
    elif args.terminate:
        terminate_pod(args.terminate)
    elif args.terminate_all:
        terminate_all()
