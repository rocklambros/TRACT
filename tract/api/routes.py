from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
import hmac 
import hashlib, json , time, uuid
from pathlib import Path
from tract.api.schemas import (
    AssignBatchRequest,
    AssignRequest,
    AssignResponse,
    BatchAssignResponse,
    BatchResultEntry,
    DuplicateMatchOut,
    DuplicateRequest,
    DuplicateResponse,
    HealthResponse,
    LivenessResponse,
    HubAssignment,
    HubDetail,
    HubListResponse,
    HubSummary,
    VersionResponse,
)
from tract.api.settings import ApiSettings, get_settings
from tract.inference import DuplicateMatch, HubPrediction, TRACTPredictor


router = APIRouter(prefix="/v1")

def _append_jsonl(path: Path, record: dict) -> None:
    """Atomic-append a JSONL record. Single write() → POSIX atomicity for small records."""
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

def require_auth(request: Request, settings: ApiSettings = Depends(get_settings)) -> None:
    expected = settings.auth_token
    if expected is None:
        return  # no auth configured — dev mode; safe only on loopback (Layer 1 enforces)
    provided = request.headers.get("X-Tract-Auth-Token", "")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="missing or invalid auth token")
    
def get_predictor(request: Request) -> TRACTPredictor:
    """Dependency: pulls the predictor off app.state"""
    return request.app.state.predictor


def _to_hub_assignment(
    pred: HubPrediction, include_raw_similarity: bool = False
) -> HubAssignment:
    """Adapter: HubPrediction (dataclass) -> HubAssignment (API response model).

    Adds the rank field (1-indexed). Drops `is_ood` (it's lifted to the
    response envelope, not per-prediction). raw_similarity is omitted unless
    the caller opted in via include_raw_similarity — it's uncalibrated and
    easy to mistake for a confidence score.
    """
    return HubAssignment(
        hub_id=pred.hub_id,
        hub_name=pred.hub_name,
        hierarchy_path=pred.hierarchy_path,
        raw_similarity=pred.raw_similarity if include_raw_similarity else None,
        calibrated_confidence=pred.calibrated_confidence,
        in_conformal_set=pred.in_conformal_set,
        rank=rank,
    )


def _to_dup_out(d: DuplicateMatch) -> DuplicateMatchOut:
    """Adapter: DuplicateMatch (dataclass) -> DuplicateMatchOut (API response model)."""
    return DuplicateMatchOut(
        control_id=d.control_id,
        framework_id=d.framework_id,
        similarity=d.similarity,
        tier=d.tier,
    )

@router.post("/assign", response_model=AssignResponse, response_model_exclude_none=True)
def assign(
    req: AssignRequest,
    request: Request,
    predictor: TRACTPredictor = Depends(get_predictor),
    settings: ApiSettings = Depends(get_settings),
    _: None = Depends(require_auth) 
) -> AssignResponse:
    """Single-text hub assignment."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    start = time.perf_counter()
    preds = predictor.predict(req.text, top_k=req.top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    
    response = AssignResponse(
        request_id=request_id,
        assignments=[
            _to_hub_assignment(p, req.include_raw_similarity) for p in preds
        ],
        t_deploy=predictor.t_deploy,
        ood_threshold=predictor.ood_threshold,
        ood_flag=preds[0].is_ood ,
        ood_score=preds[0].raw_similarity,
        model_adapter_hash=predictor.model_adapter_hash[:12],  # short SHA, deterministic from
    )
    
    if settings.audit_log_path:
        _append_jsonl(settings.audit_log_path, {
            "request_id": request_id,
            "ts": time.time(),
            "route": "/v1/assign",
            "adapter_hash": predictor.model_adapter_hash,
            "input_sha256": hashlib.sha256(req.text.encode("utf-8")).hexdigest(),
            "input_len": len(req.text),
            "top_k": req.top_k,
            "top_hub_id": preds[0].hub_id if preds else None,
            "top_confidence": preds[0].calibrated_confidence if preds else None,
            "ood_flag": preds[0].is_ood if preds else None,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    return response


@router.post("/assign/batch", response_model=BatchAssignResponse, response_model_exclude_none=True)
def assign_batch(
    req: AssignBatchRequest,
    predictor: TRACTPredictor = Depends(get_predictor),
    settings: ApiSettings = Depends(get_settings),
    _: None = Depends(require_auth) 

) -> BatchAssignResponse:
    """Batch hub assignment. Use this instead of N parallel /assign calls."""
    texts = [c.text for c in req.controls]
    ids = [c.id for c in req.controls]
    all_preds = predictor.predict_batch(texts, top_k=req.top_k)

    results: list[BatchResultEntry] = []
    for control_id, preds in zip(ids, all_preds, strict=True):
        if not preds:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"no predictions for control {control_id}",
            )
        results.append(BatchResultEntry(
            control_id=control_id,
            assignments=[_to_hub_assignment(p, i + 1) for i, p in enumerate(preds)],
            ood_flag=preds[0].is_ood,
            ood_score=preds[0].raw_similarity,
        ))

    return BatchAssignResponse(results=results, model_version=settings.model_version)


@router.post("/duplicates", response_model=DuplicateResponse)
def find_duplicates(
    req: DuplicateRequest,
    predictor: TRACTPredictor = Depends(get_predictor),
    _: None = Depends(require_auth)
) -> DuplicateResponse:
    """Find existing controls similar to the submitted text."""
    dups, sim = predictor.find_duplicates(
        req.text,
        duplicate_threshold=req.duplicate_threshold,
        similar_threshold=req.similar_threshold,
    )
    return DuplicateResponse(
        duplicates=[_to_dup_out(d) for d in dups],
        similar=[_to_dup_out(s) for s in sim],
    )


@router.get("/hubs", response_model=HubListResponse)
def list_hubs(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    predictor: TRACTPredictor = Depends(get_predictor),
    _: None = Depends(require_auth) 

) -> HubListResponse:
    """Paginated list of all CRE hubs (522 total)."""
    all_hubs = list(predictor.hierarchy.hubs.values())
    start = (page - 1) * page_size
    end = start + page_size
    hub_summaries = [
        HubSummary(
            hub_id=h.hub_id,
            name=h.name,
            hierarchy_path=h.hierarchy_path,
        )
        for h in all_hubs[start:end]
    ]
    return HubListResponse(
        hubs=hub_summaries,
        total=len(all_hubs),
        page=page,
        page_size=page_size,
    )


@router.get("/hubs/{hub_id}", response_model=HubDetail)
def get_hub(
    hub_id: str = Path(..., max_length=128, pattern=r"^[A-Za-z0-9._\-/]+$"),
    predictor: TRACTPredictor = Depends(get_predictor),
    _: None = Depends(require_auth) 

) -> HubDetail:
    """Single hub with parent + children IDs."""
    node = predictor.hierarchy.hubs.get(hub_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )
    return HubDetail(
        hub_id=hub_id,
        name=node.name,
        hierarchy_path=node.hierarchy_path,
        parent_id=node.parent_id,
        children_ids=list(node.children_ids),
    )


@router.get("/health", response_model=HealthResponse)
def health(predictor: TRACTPredictor = Depends(get_predictor), _: None = Depends(require_auth)) -> HealthResponse:
    """Readiness probe. Returns 200 only if predictor loaded and ready to serve."""
    return HealthResponse(
        status="ok",
        model_adapter_hash=predictor.model_adapter_hash,
        
    )

@router.get("/livez", response_model=LivenessResponse)
def livez(_: None = Depends(require_auth)) -> LivenessResponse:
    """Cheap liveness probe — proves the process is responding. Does NOT verify model load."""
    return LivenessResponse(status="alive")

@router.get("/version", response_model=VersionResponse)
def version(
    predictor: TRACTPredictor = Depends(get_predictor),
    settings: ApiSettings = Depends(get_settings),
    _: None = Depends(require_auth) 

) -> VersionResponse:
    """Versioning info for client cache-busting and bug reports."""
    try:
        tract_version = pkg_version("tract")
    except PackageNotFoundError:
        tract_version = "0.0.0+unknown"
    return VersionResponse(
        tract_version=tract_version,
        model_adapter_hash=predictor.model_adapter_hash,
        deployment_artifact_timestamp=predictor.deployment_timestamp,
        package_version=tract_version,
    )
