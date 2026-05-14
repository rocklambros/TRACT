from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

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
    HubAssignment,
    HubDetail,
    HubListResponse,
    HubSummary,
    VersionResponse,
)
from tract.api.settings import ApiSettings, get_settings
from tract.inference import DuplicateMatch, HubPrediction, TRACTPredictor


router = APIRouter(prefix="/v1")


def get_predictor(request: Request) -> TRACTPredictor:
    """Dependency: pulls the predictor off app.state"""
    return request.app.state.predictor


def _to_hub_assignment(pred: HubPrediction, rank: int) -> HubAssignment:
    """Adapter: HubPrediction (dataclass) -> HubAssignment (API response model).

    Adds the rank field (1-indexed). Drops `is_ood` (it's lifted to the
    response envelope, not per-prediction).
    """
    return HubAssignment(
        hub_id=pred.hub_id,
        hub_name=pred.hub_name,
        hierarchy_path=pred.hierarchy_path,
        raw_similarity=pred.raw_similarity,
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


@router.post("/assign", response_model=AssignResponse)
def assign(
    req: AssignRequest,
    predictor: TRACTPredictor = Depends(get_predictor),
    settings: ApiSettings = Depends(get_settings),
) -> AssignResponse:
    """Single-text hub assignment."""
    preds = predictor.predict(req.text, top_k=req.top_k)
    if not preds:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="no predictions returned by predictor",
        )
    assignments = [_to_hub_assignment(p, i + 1) for i, p in enumerate(preds)]
    return AssignResponse(
        assignments=assignments,
        ood_flag=preds[0].is_ood,
        ood_score=preds[0].raw_similarity,
        model_version=settings.model_version,
    )


@router.post("/assign/batch", response_model=BatchAssignResponse)
def assign_batch(
    req: AssignBatchRequest,
    predictor: TRACTPredictor = Depends(get_predictor),
    settings: ApiSettings = Depends(get_settings),
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
    hub_id: str,
    predictor: TRACTPredictor = Depends(get_predictor),
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
def health(predictor: TRACTPredictor = Depends(get_predictor)) -> HealthResponse:
    """Liveness probe. Returns 200 only if predictor loaded."""
    return HealthResponse(
        status="ok",
        model_adapter_hash=predictor.model_adapter_hash,
        t_deploy=predictor.t_deploy,
        ood_threshold=predictor.ood_threshold,
    )


@router.get("/version", response_model=VersionResponse)
def version(
    predictor: TRACTPredictor = Depends(get_predictor),
    settings: ApiSettings = Depends(get_settings),
) -> VersionResponse:
    """Versioning info for client cache-busting and bug reports."""
    try:
        tract_version = pkg_version("tract")
    except PackageNotFoundError:
        tract_version = "0.0.0+unknown"
    return VersionResponse(
        tract_version=tract_version,
        model_version=settings.model_version,
        model_adapter_hash=predictor.model_adapter_hash,
        deployment_artifact_timestamp=predictor.deployment_timestamp,
    )
