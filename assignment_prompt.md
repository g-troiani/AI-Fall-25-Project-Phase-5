Project 5 – Final Integrated System (Weeks 11–14)

Goal: integrate everything + add generative, graph, reinforcement learning, and MLOps/Ethics.

Integrations to include:

Regression/Classification (from Projects 1–2)
Neural Networks (Project 3)
Sequential/Transformer (Project 4)
Generative or GNN component (or both)
Reinforcement Learning or MLOps (or both)
Do this:

Generative: produce synthetic samples to augment training (provided VAE-style NumPy sampler).
Graph: simple link prediction/recommendation from bipartite edges (customers–products) using common-neighbor scoring.
RL: Q-learning on a small gridworld for decision policies (e.g., routing reward).
MLOps/Ethics: capture metrics to a CSV “run log”; discuss bias sources, privacy, and reproducibility.
What I gave you:

Data project5_final_system/data/
graph_edges.csv  Download graph_edges.csv(toy customer–product edges)
gridworld.csv Download gridworld.csv(5×5 reward map)
Starter/model code project5_final_system/src/
vae_synth.py Download vae_synth.py(NumPy latent sampler placeholder)
gnn_link_pred.py Download gnn_link_pred.py(common-neighbors link scoring)
q_learning.py Download q_learning.py(tabular Q-learning)
eval_pipeline.py Download eval_pipeline.py(lightweight metrics logger CSV)
Docs/Reports templates
Repository Layout (expanded)

project5_final_system/

├─ data/

│  ├─ base/                         # immutable inputs checked into git (small/toy)

│  │  ├─ graph_edges.csv            # customer,product edges

│  │  └─ gridworld.csv              # 5x5 rewards

│  └─ derived/                      # generated/augmented datasets (gitignored)

├─ src/

│  ├─ classic/                      # P1–P2 models

│  │  ├─ train_tabular.py

│  │  └─ infer_tabular.py

│  ├─ neural/                       # P3 MLP/CNN/RNN

│  │  ├─ train_mlp.py

│  │  ├─ train_vision_cnn.py

│  │  └─ train_text_lstm.py

│  ├─ sequential/                   # P4 LSTM/Transformer

│  │  ├─ train_timeseries.py

│  │  └─ train_text_transformer.py

│  ├─ generative/

│  │  └─ vae_synth.py               # provided NumPy latent sampler placeholder

│  ├─ graph/

│  │  └─ gnn_link_pred.py           # common-neighbor scorer (baseline)

│  ├─ rl/

│  │  └─ q_learning.py              # tabular Q-learning for gridworld

│  ├─ mlops/

│  │  ├─ eval_pipeline.py           # provided CSV logger

│  │  ├─ runlog_schema.json         # schema for run_log.csv columns

│  │  └─ utils.py                   # seed, hashing, config load, timer

│  ├─ api/

│  │  └─ service.py                 # FastAPI/CLI facade that routes requests to modules

│  └─ integration/

│     └─ orchestrate.py             # end-to-end DAG / CLI entrypoint

├─ outputs/

│  ├─ models/                       # serialized weights (gitignored)

│  ├─ artifacts/                    # plots, confusion matrices, attention maps

│  └─ run_log.csv                   # all experiments (append-only)

├─ configs/

│  ├─ default.yaml                  # global config (paths, seeds)

│  ├─ classic.yaml

│  ├─ neural.yaml

│  ├─ sequential.yaml

│  ├─ generative.yaml

│  ├─ graph.yaml

│  └─ rl.yaml

├─ docs/

│  ├─ Final_Report_Template.docx

│  ├─ Slides_Template.pptx

│  ├─ System_Diagram.png            # your architecture diagram

│  └─ Model_Cards/                  # one-page cards per component

├─ tests/

│  ├─ test_data_contracts.py

│  ├─ test_api_contract.py

│  └─ test_smoke_integration.py

├─ README.md

└─ requirements.txt

System Scope (what to integrate)

Regression/Classification (P1–P2)
Reuse your best classic baseline (e.g., Logistic Regression / RandomForest).
Expose fit/predict/proba via a slim adapter so it plugs into py.
Neural Networks (P3)
Choose one strong neural model from P3 (MLP for tabular, CNN for vision, or LSTM for text).
Add MC Dropout inference AND calibration (temp scaling optional).
Sequential/Transformer (P4)
Include either time-series forecaster (LSTM/Transformer) or text Transformer classifier—preferably both if feasible.
Provide windowing/tokenization utilities under sequential/.
Generative OR Graph (or both)
Generative augmentation: use py to sample N synthetic rows, then ablation-compare training with/without augmentation on your P3/P4 model.
Graph link prediction: from csv, compute Common Neighbors, Jaccard, or Adamic–Adar scores to recommend K products for a customer. (PyG optional; baseline is fine.)
Reinforcement Learning OR MLOps (or both)
RL: implement tabular Q-learning on csv (γ∈[0.9,0.99], ε-greedy, α decay). Save learned Q, policy heatmap, and average episodic return curve.
MLOps: use py to append runs to outputs/run_log.csv with ALL of:
run_id, timestamp, git_commit, data_hash, config_hash, seed, component, metric_name, metric_value, latency_ms, params_json, notes.
✅ Minimum: Items 1–3 + one of (4) or (5).
🌟 Target/Recommended: Items 1–5 (all).
🏅 Stretch: Both Generative and Graph, plus both RL and MLOps extras (model cards, drift check, simple CI).

Orchestration Pattern (how things talk)

DAG in integration/orchestrate.py:
load_config() and set seed.
load_data() (tabular/vision/text/timeseries).
Optional: augment = vae_synth.sample(n) → concat(train, augment).
Train classic and neural baselines 👉 log metrics.
If sequential enabled: train TS forecaster or text transformer 👉 log horizon/classification metrics.
If graph enabled: build recommender 👉 log MAP@K / Hit@K on masked edges.
If RL enabled: train Q-agent 👉 log episodic returns and convergence time.
Register artifacts (paths) and append run rows to csv.
Export system diagram & comparison tables to docs/.
API façade in api/service.py (choose CLI or FastAPI):
POST /predict/tabular → classic/neural ensemble (optional soft-vote).
POST /recommend/:customer_id → graph scorer top-K.
POST /forecast/ts → sequential forecaster horizon H.
GET /policy/gridworld → Q-policy grid (JSON).
Metrics (what to measure)

Supervised (tabular/vision/text)

Accuracy, F1-macro, ROC-AUC (if binary), ECE, latency (ms/inference, batch=1 and batch=32).
With/without generative augmentation (report deltas).
Time Series

MAE, MAPE, and MASE over horizon H (e.g., 14 or 30).
Plot true vs. predicted, residuals, prediction intervals via MC Dropout.
Graph Recommendation

Split by edge hiding: mask 20% edges per customer as test.
Report Hit@K, MAP@K, and coverage.
Reinforcement Learning

Average episodic return vs. episode curve; policy visualization.
Convergence time (episodes) and sensitivity to ε/α schedules.
MLOps Logging (mandatory if MLOps chosen)

Append every experiment line to outputs/run_log.csv. Schema:

run_id,timestamp,git_commit,data_hash,config_hash,seed,component,metric_name,metric_value,latency_ms,params_json,notes

data_hash: SHA256 of input file(s)
config_hash: SHA256 of YAML used
component: classic|neural|sequential|graph|rl
params_json: serialized key hyperparams (keep < 256 chars)
Ethics, Risk & Reproducibility (explicit section in report)

Address at minimum:

Bias sources (sampling, label leakage, augmentation artifacts).
Privacy (data retention; how synthetic data is labeled/flagged; no re-identification).
Transparency (model cards with intended use, limits, metrics).
Reproducibility (seed control, config files, pinned deps, data immutability).
Safety (confidence thresholds, human-in-the-loop for low-confidence cases).
Provide a 1-page Model Card per major component under docs/Model_Cards/.

Implementation Details & Hints

Generative Augmentation (VAE-style sampler)

Inputs: X_train normalized; sampler returns X_synth ~ pθ(x).
Ablation: Train neural model with {0%, 10%, 25%} synthetic mix.
Guardrails: KL-filter out outliers; cap per-class synth at minority * 1.0.
Graph Link Prediction (baseline)

Build customer→neighbors and product→neighbors maps.
Common Neighbors score for (c,p):



Add Jaccard and Adamic–Adar as optional scorers; compare Hit@K.
Q-Learning (gridworld)

Tabular Q[s, a]; ε-greedy; α_t = α₀/(1+t/τ).
Log return_per_episode and export policy_grid.png.
Latency & Context (summary in report)

Measure: time.perf_counter() around model(x) for 100 runs; report mean±std.
Discuss quadratic attention cost vs. recurrent costs; justify chosen sequence length.
Minimal Make/CLI

# classic + neural

python src/integration/orchestrate.py --stages classic,neural --config configs/default.yaml

# add augmentation and TS forecast

python src/integration/orchestrate.py --stages generative,neural,sequential

# add graph + RL

python src/integration/orchestrate.py --stages graph,rl

Artifacts to Produce

System Diagram (docs/System_Diagram.png)
Boxes: Data → Augment (VAE) → Supervised (Classic/Neural) → Sequential (TS/NLP) → Graph Recommender → RL Policy
Sidecar: MLOps Logger (captures metrics, configs, artifacts).
Key Plots (save under outputs/artifacts/)
Confusion matrices, reliability diagrams, TS forecasts (with intervals), graph Hit@K bars, RL policy heatmap & return curve.
Comparison Tables
Augmentation deltas; Graph scorer comparison; RL hyperparam sensitivity.
Final Deliverables (Week 14)

A) 10–15 min Presentation
1 slide: Problem & Data (what your integrated system does).
2 slides: Architecture & Orchestration (diagram + data flow).
2 slides: Results (supervised, TS, graph, RL highlights).
1 slide: MLOps & Ethics (run log, model cards, risks & mitigations).
1 slide: Lessons Learned & Next Steps.
1 slide: Live Demo (CLI/API call showing end-to-end).
B) Final Report (8–12 pages)
Abstract & Goals (½ page)
Data & Preprocessing (1 page)
Methods (2–3 pages; per component + orchestration)
Results (2–3 pages; metrics, plots, ablations)
MLOps & Ethics (1–2 pages; logging, cards, risks)
Discussion & Limitations (1 page)
Appendix (run log snippet, configs, seeds)
C) Code Repo
README.md with quickstart, environment, commands, and component map.
requirements.txt pinned (e.g., torch, sklearn, pandas, matplotlib, fastapi or click, pydantic).
outputs/run_log.csv with ≥ 10 appended runs spanning modules.
docs/Model_Cards/*.md (at least supervised+sequential+graph or rl).
Starter Snippets (concise)

Run Logger (append-only)

# src/mlops/utils.py

import csv, hashlib, json, subprocess, time, pathlib

def sha256(path):

    h=hashlib.sha256(); h.update(pathlib.Path(path).read_bytes()); return h.hexdigest()

def git_commit():

    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()

    except: return "nogit"

def log_row(runlog, **kw):

    file_exists = pathlib.Path(runlog).exists()

    with open(runlog, "a", newline="") as f:

        w = csv.DictWriter(f, fieldnames=[

          "run_id","timestamp","git_commit","data_hash","config_hash","seed",

          "component","metric_name","metric_value","latency_ms","params_json","notes"])

        if not file_exists: w.writeheader()

        w.writerow(kw)

Common-Neighbors (mask-then-score)

# src/graph/gnn_link_pred.py (baseline)

from collections import defaultdict

def common_neighbors(train_edges):

    c2p = defaultdict(set); p2c = defaultdict(set)

    for c,p in train_edges:

        c2p[c].add(p); p2c[p].add(c)

    def score(c, p): return len(c2p[c] & {q for q in p2c[p] for _ in [0]})

    return score

Q-Learning Loop (gridworld)

# src/rl/q_learning.py (core idea)

for ep in range(episodes):

    s = env.reset(); done = False; ret = 0

    while not done:

        a = epsilon_greedy(Q[s], eps)

        s2, r, done = env.step(a)

        Q[s][a] += alpha * (r + gamma * max(Q[s2]) - Q[s][a])

        s = s2; ret += r

    returns.append(ret); eps = max(eps_min, eps*eps_decay)

Submission Checklist (quick)

End-to-end script runs from a clean clone with a fresh virtualenv.
outputs/run_log.csv includes ≥10 runs across components.
System diagram + required plots saved to outputs/artifacts/.
Final report (8–12 pages) & 10–15 min deck in docs/.
Model cards present; risks & mitigations documented.
Tests pass (pytest -q), or provide a short note on any skipped tests.
Evaluation Rubric (100 pts)

Area

Pts

What we’re looking for

System Integration & Orchestration

20

Clean DAG, modular adapters, consistent IO contracts, end-to-end run.

Supervised + Neural (baseline & P3)

15

Correct training/eval, calibration or intervals, clear gains vs classic.

Sequential/Transformer (P4)

15

Proper windowing/tokenization, robust metrics, plots.

Generative OR Graph (min)

10

Working augmentation or recommender with metrics.

RL OR MLOps (min)

10

Converging Q-policy or complete run logging with schema compliance.

Reproducibility & MLOps Hygiene

10

Seeds, config files, pinned deps, run log completeness, artifact paths.

Ethics & Model Cards

10

Concrete risks/mitigations, clear model cards, privacy stance.

Analysis & Communication

10

Insightful comparisons, ablations, latency/context trade-offs.

Bonus (up to +10): implement both (Generative & Graph) and both (RL & MLOps), plus a tiny CI (lint + pytest -q) and a smoke integration test.