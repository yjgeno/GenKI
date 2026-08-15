"""Pydantic request/response models for the GenKI web API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    n_genes: int
    n_cells: int
    gene_names: list[str]
    obs_labels: dict[str, list[str]] = Field(
        default_factory=dict,
        description="obs column name -> sorted unique values, for columns usable as target_cell filters",
    )


class JobCreate(BaseModel):
    dataset_id: str
    target_gene: list[str] = Field(..., min_length=1)
    target_cell: Optional[str] = None
    obs_label: str = "ident"
    epochs: int = Field(100, ge=1, le=5000)
    lr: float = Field(7e-4, gt=0)
    seed: Optional[int] = None
    n_permutations: int = Field(100, ge=0, le=2000)
    by: str = "KL"
    n_cpus: int = Field(
        -1,
        description=(
            "parallel workers for GRN construction; -1 (default) uses all local "
            "CPUs. Requires the ray extra, bundled with GenKI[web]; falls back "
            "to a single process if ray isn't installed."
        ),
    )

    @field_validator("n_cpus")
    @classmethod
    def _n_cpus_valid(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError("n_cpus must be -1 (all local CPUs) or a positive integer")
        return v


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    stage: str
    error: Optional[str] = None


class GeneRankRow(BaseModel):
    gene: str
    dis: float
    rank: Optional[int] = None
    hit: Optional[int] = None


class JobResult(BaseModel):
    job_id: str
    target_gene: list[str]
    rows: list[GeneRankRow]
    metrics: Optional[list] = None
