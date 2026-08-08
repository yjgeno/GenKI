"""FastAPI app for the GenKI local web UI.

Route layout:
    GET  /api/datasets/example       load the bundled example dataset
    POST /api/datasets               upload a .h5ad dataset
    POST /api/jobs                   start a knock-out run
    GET  /api/jobs/{id}              poll job status/stage
    GET  /api/jobs/{id}/result       ranked genes as JSON
    GET  /api/jobs/{id}/result.csv   ranked genes as a CSV download
    GET  /                           static single-page app
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from anndata import AnnData
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from GenKI.preprocessing import build_adata

from .jobs import DatasetNotFoundError, JobManager, JobNotFoundError
from .schemas import (
    DatasetInfo,
    GeneRankRow,
    JobCreate,
    JobCreated,
    JobResult,
    JobStatus,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def _find_example_dataset() -> Optional[Path]:
    """Locate the bundled example .h5ad, if available.

    The 124 MB example file (data/microglial_seurat_WT.h5ad) is not shipped
    inside the pip wheel (it's excluded from package data), so this only
    resolves when running out of a git checkout — set GENKI_EXAMPLE_DATA to
    override, otherwise upload a dataset instead.
    """
    env_path = os.environ.get("GENKI_EXAMPLE_DATA")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    # repo checkout, e.g. `pip install -e .` or running from a clone
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "microglial_seurat_WT.h5ad")
    candidates.append(Path.cwd() / "data" / "microglial_seurat_WT.h5ad")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _dataset_info(dataset_id: str, name: str, adata: AnnData) -> DatasetInfo:
    obs_labels: dict[str, list[str]] = {}
    for col in adata.obs.columns:
        try:
            values = adata.obs[col].unique()
        except TypeError:
            continue
        if 1 < len(values) <= 50:
            obs_labels[col] = sorted(str(v) for v in values)
    return DatasetInfo(
        dataset_id=dataset_id,
        name=name,
        n_genes=adata.n_vars,
        n_cells=adata.n_obs,
        gene_names=list(adata.var_names),
        obs_labels=obs_labels,
    )


def create_app(grn_dir: str = "GRNs") -> FastAPI:
    app = FastAPI(title="GenKI", description="Local UI for GenKI knock-out inference")
    manager = JobManager(grn_dir=grn_dir)

    # -- datasets -----------------------------------------------------
    @app.get("/api/datasets/example", response_model=DatasetInfo)
    def load_example_dataset():
        path = _find_example_dataset()
        if path is None:
            raise HTTPException(
                404,
                "bundled example dataset not found (only available from a git "
                "checkout) — upload your own .h5ad instead",
            )
        adata = build_adata(str(path))
        dataset_id = manager.add_dataset(adata)
        return _dataset_info(dataset_id, path.name, adata)

    @app.post("/api/datasets", response_model=DatasetInfo)
    async def upload_dataset(file: UploadFile):
        if not file.filename.endswith(".h5ad"):
            raise HTTPException(400, "only .h5ad files are supported")

        tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        try:
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large (limit 500 MB)")
                tmp.write(chunk)
            tmp.close()
            try:
                adata = build_adata(tmp.name)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"could not read .h5ad file: {exc}") from exc
        finally:
            tmp.close()
            os.unlink(tmp.name)

        dataset_id = manager.add_dataset(adata)
        return _dataset_info(dataset_id, file.filename, adata)

    # -- jobs -----------------------------------------------------------
    @app.post("/api/jobs", response_model=JobCreated)
    def create_job(params: JobCreate):
        try:
            adata = manager.get_dataset(params.dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(404, f"unknown dataset_id {params.dataset_id!r}") from exc

        gene_names = set(adata.var_names)
        missing = [g for g in params.target_gene if g not in gene_names]
        if missing:
            raise HTTPException(400, f"target gene(s) not found in dataset: {missing}")

        job_id = manager.submit(params)
        return JobCreated(job_id=job_id)

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def get_job_status(job_id: str):
        try:
            job = manager.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(404, f"unknown job_id {job_id!r}") from exc
        return JobStatus(**job.to_status_dict())

    @app.get("/api/jobs/{job_id}/result", response_model=JobResult)
    def get_job_result(job_id: str):
        job = _require_finished_job(manager, job_id)
        df = job.result
        rows = [
            GeneRankRow(
                gene=str(gene),
                dis=float(row["dis"]),
                rank=int(row["rank"]) if "rank" in row else None,
                hit=int(row["hit"]) if "hit" in row else None,
            )
            for gene, row in df.iterrows()
        ]
        return JobResult(job_id=job_id, target_gene=job.params.target_gene, rows=rows)

    @app.get("/api/jobs/{job_id}/result.csv")
    def get_job_result_csv(job_id: str):
        job = _require_finished_job(manager, job_id)
        buf = io.StringIO()
        job.result.to_csv(buf, index_label="gene")
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="genki_{job_id}.csv"'}
        return StreamingResponse(buf, media_type="text/csv", headers=headers)

    # -- static frontend --------------------------------------------
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def _require_finished_job(manager: JobManager, job_id: str):
    try:
        job = manager.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, f"unknown job_id {job_id!r}") from exc
    if job.status == "error":
        raise HTTPException(500, f"job failed: {job.error}")
    if job.status != "done":
        raise HTTPException(409, f"job not finished yet (status={job.status}, stage={job.stage})")
    return job


app = create_app()
