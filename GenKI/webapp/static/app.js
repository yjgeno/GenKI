"use strict";

const state = {
  dataset: null, // {dataset_id, name, n_genes, n_cells, gene_names, obs_labels}
  targetGenes: [],
  resultRows: null,
  sort: { key: "rank", dir: 1 },
};

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      /* ignore non-JSON error bodies */
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

function showError(el, message) {
  el.textContent = message;
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}

// -- dataset ------------------------------------------------------------

$("use-example").addEventListener("click", async () => {
  hide($("dataset-error"));
  $("use-example").disabled = true;
  try {
    const info = await api("/api/datasets/example");
    setDataset(info);
  } catch (err) {
    showError($("dataset-error"), err.message);
  } finally {
    $("use-example").disabled = false;
  }
});

$("file-input").addEventListener("change", async (evt) => {
  const file = evt.target.files[0];
  if (!file) return;
  hide($("dataset-error"));
  const form = new FormData();
  form.append("file", file);
  try {
    const info = await api("/api/datasets", { method: "POST", body: form });
    setDataset(info);
  } catch (err) {
    showError($("dataset-error"), err.message);
  } finally {
    evt.target.value = "";
  }
});

function setDataset(info) {
  state.dataset = info;
  state.targetGenes = [];
  renderChips();

  const geneList = $("gene-list");
  geneList.innerHTML = "";
  for (const gene of info.gene_names) {
    const opt = document.createElement("option");
    opt.value = gene;
    geneList.appendChild(opt);
  }

  const obsCols = Object.keys(info.obs_labels || {});
  const obsField = $("obs-label-field");
  const obsSelect = $("obs-label-select");
  if (obsCols.length > 0) {
    obsSelect.innerHTML = "";
    for (const col of obsCols) {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      obsSelect.appendChild(opt);
    }
    obsField.hidden = false;
    populateTargetCellOptions();
  } else {
    obsField.hidden = true;
    $("target-cell-field").hidden = true;
  }

  const infoEl = $("dataset-info");
  infoEl.textContent =
    `Loaded "${info.name}" — ${info.n_genes.toLocaleString()} genes × ${info.n_cells.toLocaleString()} cells.`;
  infoEl.classList.remove("hidden");

  $("run-section").classList.remove("hidden");
  $("results-section").classList.add("hidden");
}

$("obs-label-select").addEventListener("change", populateTargetCellOptions);

function populateTargetCellOptions() {
  const col = $("obs-label-select").value;
  const values = (state.dataset.obs_labels || {})[col] || [];
  const select = $("target-cell-select");
  select.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "All cells";
  select.appendChild(allOpt);
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  }
  $("target-cell-field").hidden = false;
}

// -- target gene picker ---------------------------------------------

// Returns true if `raw` was a valid, newly-added (or already-added) gene.
function tryAddGene(raw) {
  raw = raw.trim().toUpperCase();
  if (!raw) return false;
  if (!state.dataset.gene_names.includes(raw)) {
    showError($("dataset-error"), `"${raw}" is not a gene in the loaded dataset`);
    return false;
  }
  hide($("dataset-error"));
  if (!state.targetGenes.includes(raw)) {
    state.targetGenes.push(raw);
    renderChips();
  }
  return true;
}

$("target-gene-input").addEventListener("keydown", (evt) => {
  if (evt.key !== "Enter") return;
  evt.preventDefault();
  if (tryAddGene(evt.target.value)) evt.target.value = "";
});

// Clicking an option in the native <datalist> dropdown only changes the
// input's value (no Enter keydown fires), so commit it as soon as the
// value exactly matches a known gene — covers mouse/touch selection.
$("target-gene-input").addEventListener("input", (evt) => {
  const raw = evt.target.value.trim().toUpperCase();
  if (state.dataset && state.dataset.gene_names.includes(raw)) {
    tryAddGene(raw);
    evt.target.value = "";
  }
});

function renderChips() {
  const wrap = $("target-gene-chips");
  wrap.innerHTML = "";
  for (const gene of state.targetGenes) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = gene;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "×";
    removeBtn.setAttribute("aria-label", `remove ${gene}`);
    removeBtn.addEventListener("click", () => {
      state.targetGenes = state.targetGenes.filter((g) => g !== gene);
      renderChips();
    });
    chip.appendChild(removeBtn);
    wrap.appendChild(chip);
  }
}

// -- run + poll -----------------------------------------------------

$("run-form").addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const geneInput = $("target-gene-input");
  if (geneInput.value.trim() && tryAddGene(geneInput.value)) {
    geneInput.value = "";
  }
  if (state.targetGenes.length === 0) {
    showError($("dataset-error"), "add at least one target gene");
    return;
  }
  hide($("dataset-error"));

  const obsField = $("obs-label-field");
  const payload = {
    dataset_id: state.dataset.dataset_id,
    target_gene: state.targetGenes,
    target_cell: obsField.hidden ? null : $("target-cell-select").value || null,
    obs_label: obsField.hidden ? "ident" : $("obs-label-select").value,
    epochs: Number($("epochs").value),
    lr: Number($("lr").value),
    seed: $("seed").value === "" ? null : Number($("seed").value),
    n_permutations: Number($("n-permutations").value),
    by: $("by").value,
  };

  $("run-button").disabled = true;
  $("results-section").classList.remove("hidden");
  hide($("results-error"));
  $("results-table-wrap").innerHTML = "";
  hide($("download-csv"));
  $("status-bar").textContent = "submitting…";

  try {
    const { job_id } = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await pollJob(job_id);
  } catch (err) {
    $("status-bar").textContent = "";
    showError($("results-error"), err.message);
  } finally {
    $("run-button").disabled = false;
  }
});

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      let status;
      try {
        status = await api(`/api/jobs/${jobId}`);
      } catch (err) {
        reject(err);
        return;
      }
      if (status.status === "error") {
        reject(new Error(status.error || "job failed"));
        return;
      }
      $("status-bar").textContent = `${status.stage}…`;
      if (status.status === "done") {
        try {
          await loadResult(jobId);
          resolve();
        } catch (err) {
          reject(err);
        }
        return;
      }
      setTimeout(tick, 1000);
    };
    tick();
  });
}

async function loadResult(jobId) {
  const result = await api(`/api/jobs/${jobId}/result`);
  $("status-bar").textContent = `done — ${result.rows.length} genes ranked`;
  state.resultRows = result.rows;
  state.targetGeneSet = new Set(result.target_gene);
  state.sort = { key: "rank", dir: 1 };
  renderResultsTable();

  const link = $("download-csv");
  link.href = `/api/jobs/${jobId}/result.csv`;
  link.textContent = "Download CSV";
  link.classList.remove("hidden");
}

function renderResultsTable() {
  const rows = [...state.resultRows];
  const { key, dir } = state.sort;
  rows.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return dir * av.localeCompare(bv);
    return dir * (av - bv);
  });

  const columns = [
    { key: "rank", label: "Rank" },
    { key: "gene", label: "Gene" },
    { key: "dis", label: "Distance" },
    { key: "hit", label: "Hits" },
  ];

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col.label + (state.sort.key === col.key ? (dir === 1 ? " ▲" : " ▼") : "");
    th.addEventListener("click", () => {
      state.sort =
        state.sort.key === col.key ? { key: col.key, dir: -dir } : { key: col.key, dir: 1 };
      renderResultsTable();
    });
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (state.targetGeneSet.has(row.gene)) tr.className = "target-row";
    for (const col of columns) {
      const td = document.createElement("td");
      const v = row[col.key];
      td.textContent = v == null ? "–" : col.key === "dis" ? v.toFixed(4) : v;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  const wrap = $("results-table-wrap");
  wrap.innerHTML = "";
  wrap.appendChild(table);
}
