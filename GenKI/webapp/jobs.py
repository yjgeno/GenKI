"""In-process job manager for the GenKI web UI.

Training + permutation testing are synchronous and CPU/GPU-bound (see
GenKI/train.py), so each run is executed on a single-worker background
thread pool and tracked by job id. This is a local, single-user tool: jobs
are intentionally serialized (VGAE_trainer mutates instance state, so two
runs must not share a model concurrently) and everything lives in memory —
no persistence across process restarts.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from anndata import AnnData

from GenKI import GenKI as GenKICore

from .schemas import JobCreate

logger = logging.getLogger(__name__)

# Stages surfaced to the UI as a simple progress stepper.
STAGE_QUEUED = "queued"
STAGE_BUILDING_GRN = "building GRN"
STAGE_TRAINING = "training"
STAGE_NULL_DIST = "computing null distribution"
STAGE_RANKING = "ranking genes"
STAGE_DONE = "done"


@dataclass
class Job:
    id: str
    dataset_id: str
    params: JobCreate
    status: str = "queued"  # queued | running | done | error
    stage: str = STAGE_QUEUED
    error: Optional[str] = None
    result: Optional[pd.DataFrame] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.stage = stage

    def to_status_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
            }


class DatasetNotFoundError(KeyError):
    pass


class JobNotFoundError(KeyError):
    pass


class JobManager:
    """Holds loaded datasets and runs GenKI jobs on a single worker thread."""

    def __init__(self, grn_dir: str = "GRNs"):
        self._datasets: dict[str, AnnData] = {}
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="genki-job")
        self._grn_dir = grn_dir

    # -- datasets ---------------------------------------------------
    def add_dataset(self, adata: AnnData) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        self._datasets[dataset_id] = adata
        return dataset_id

    def get_dataset(self, dataset_id: str) -> AnnData:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise DatasetNotFoundError(dataset_id) from exc

    # -- jobs ---------------------------------------------------------
    def submit(self, params: JobCreate) -> str:
        adata = self.get_dataset(params.dataset_id)  # fail fast if unknown dataset
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dataset_id=params.dataset_id, params=params)
        self._jobs[job_id] = job
        self._executor.submit(self._run, job, adata)
        return job_id

    def get_job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def _run(self, job: Job, adata: AnnData) -> None:
        job.status = "running"
        params = job.params
        try:
            job.set_stage(STAGE_BUILDING_GRN)
            gk = GenKICore(
                adata,
                target_gene=params.target_gene,
                target_cell=params.target_cell,
                obs_label=params.obs_label,
                grn_dir=self._grn_dir,
                rebuild_grn=True,
            )

            job.set_stage(STAGE_TRAINING)
            gk.fit(epochs=params.epochs, lr=params.lr, seed=params.seed, verbose=False)

            job.set_stage(STAGE_NULL_DIST if params.n_permutations else STAGE_RANKING)
            df = gk.predict(n_permutations=params.n_permutations, by=params.by)

            job.result = df
            job.set_stage(STAGE_DONE)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("job %s failed", job.id)
            job.error = str(exc)
            job.status = "error"
