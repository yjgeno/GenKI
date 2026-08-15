"""End-to-end tests for the local FastAPI web UI (GenKI/webapp)."""

import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from GenKI.webapp.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(grn_dir=str(tmp_path / "GRNs"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def small_h5ad_path(small_adata, tmp_path):
    """The (raw, unprocessed) sample AnnData written to a temp .h5ad file.

    Mirrors README's from_h5ad flow: the webapp's upload endpoint runs
    build_adata() itself, so the uploaded file should be raw, matching
    test_api.py's use of ``small_adata`` with ``preprocess=True``.
    """
    path = tmp_path / "small.h5ad"
    small_adata.write_h5ad(path)
    return path


def _upload(client, path):
    with open(path, "rb") as fh:
        resp = client.post(
            "/api/datasets", files={"file": ("small.h5ad", fh, "application/octet-stream")}
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _run_job_to_completion(client, dataset, target_gene, **overrides):
    payload = {
        "dataset_id": dataset["dataset_id"],
        "target_gene": [target_gene],
        "epochs": 2,
        "n_permutations": 0,
        "seed": 0,
        **overrides,
    }
    resp = client.post("/api/jobs", json=payload)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + 60
    status = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert status is not None and status["status"] == "done", status
    return job_id


def test_upload_dataset_returns_shape_and_genes(client, small_h5ad_path, small_adata):
    info = _upload(client, small_h5ad_path)
    assert info["n_genes"] == small_adata.n_vars
    assert info["n_cells"] == small_adata.n_obs
    assert len(info["gene_names"]) == small_adata.n_vars


def test_upload_rejects_oversized_content_length(client):
    """A too-large Content-Length is rejected before the body is read.

    Sends a tiny body with a spoofed huge Content-Length header: if this
    passed, it'd mean the middleware read the (nonexistent) rest of the
    body rather than rejecting from the header alone, i.e. the whole point
    of checking Content-Length up front (skip parsing multi-GB uploads)
    would be broken.
    """
    from GenKI.webapp.main import MAX_UPLOAD_BYTES

    resp = client.post(
        "/api/datasets",
        content=b"not actually a big file",
        headers={
            "content-length": str(MAX_UPLOAD_BYTES + 1),
            "content-type": "multipart/form-data; boundary=x",
        },
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_upload_rejects_non_h5ad(client, tmp_path):
    bogus = tmp_path / "not_h5ad.txt"
    bogus.write_text("nope")
    with open(bogus, "rb") as fh:
        resp = client.post("/api/datasets", files={"file": ("not_h5ad.txt", fh, "text/plain")})
    assert resp.status_code == 400


def test_run_job_end_to_end(client, small_h5ad_path, small_adata):
    dataset = _upload(client, small_h5ad_path)
    target = str(small_adata.var_names[0]).upper()  # build_adata upper-cases gene names

    job_id = _run_job_to_completion(client, dataset, target)

    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["target_gene"] == [target]
    assert len(result["rows"]) == dataset["n_genes"]
    genes = {row["gene"] for row in result["rows"]}
    assert genes == set(dataset["gene_names"])
    ranks = sorted(row["rank"] for row in result["rows"])
    assert ranks == list(range(1, dataset["n_genes"] + 1))

    csv_resp = client.get(f"/api/jobs/{job_id}/result.csv")
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert "gene,dis" in csv_resp.text.splitlines()[0]


@pytest.mark.parametrize("bad_n_cpus", [0, -2, 1.5])
def test_invalid_n_cpus_rejected(client, small_h5ad_path, bad_n_cpus):
    dataset = _upload(client, small_h5ad_path)
    resp = client.post(
        "/api/jobs",
        json={
            "dataset_id": dataset["dataset_id"],
            "target_gene": [dataset["gene_names"][0]],
            "epochs": 2,
            "n_permutations": 0,
            "n_cpus": bad_n_cpus,
        },
    )
    assert resp.status_code == 422, resp.text


def test_run_job_with_explicit_n_cpus(client, small_h5ad_path, small_adata):
    """n_cpus reaches make_pcNet and doesn't disrupt a normal run.

    Ray may or may not be installed in the test environment; either way
    pcNet.make_pcNet degrades gracefully (see test_pcnet.py), so this just
    checks the job still completes and returns a full ranking.
    """
    dataset = _upload(client, small_h5ad_path)
    target = str(small_adata.var_names[0]).upper()

    job_id = _run_job_to_completion(client, dataset, target, n_cpus=2)

    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert len(result["rows"]) == dataset["n_genes"]


def test_invalid_target_gene_rejected(client, small_h5ad_path):
    dataset = _upload(client, small_h5ad_path)
    resp = client.post(
        "/api/jobs",
        json={
            "dataset_id": dataset["dataset_id"],
            "target_gene": ["NOT_A_REAL_GENE"],
            "epochs": 2,
            "n_permutations": 0,
        },
    )
    assert resp.status_code == 400


def test_unknown_dataset_id_rejected(client):
    resp = client.post(
        "/api/jobs",
        json={"dataset_id": "does-not-exist", "target_gene": ["X"], "epochs": 2},
    )
    assert resp.status_code == 404


def test_unknown_job_id_returns_404(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
    resp = client.get("/api/jobs/does-not-exist/result")
    assert resp.status_code == 404


def test_example_dataset_when_available(client):
    """Only meaningful in a git checkout that has data/microglial_seurat_WT.h5ad."""
    resp = client.get("/api/datasets/example")
    if resp.status_code == 404:
        pytest.skip("bundled example dataset not present (not shipped in the wheel)")
    assert resp.status_code == 200
    info = resp.json()
    assert info["n_genes"] > 0
    assert info["n_cells"] > 0
