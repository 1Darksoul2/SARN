# SARN — Self-Attention Reconstruction Network for Aviation Anomaly Detection

> **Real-time, unsupervised anomaly detection for turbofan jet engines using a Transformer-based autoencoder trained on NASA C-MAPSS data.**

---

## Table of Contents

1. [What is this system?](#what-is-this-system)
2. [System Architecture](#system-architecture)
3. [How the ML Model Works (SARN)](#how-the-ml-model-works-sarn)
4. [Quick Start](#quick-start)
5. [Running the Full Offline Pipeline](#running-the-full-offline-pipeline)
6. [Running the Real-Time Live Dashboard](#running-the-real-time-live-dashboard)
7. [Running the External Telemetry Simulator](#running-the-external-telemetry-simulator)
8. [All Pipeline Arguments](#all-pipeline-arguments)
9. [Understanding the Output](#understanding-the-output)
10. [LangGraph Diagnostic Agent](#langgraph-diagnostic-agent)

---

## What is this System?

This project detects anomalous behavior in aircraft jet engines in real time. It operates in two modes:

| Mode | Script | Use Case |
|---|---|---|
| **Offline Pipeline** | `run_pipeline.py` | Train the model on historical NASA C-MAPSS data, inject synthetic faults, evaluate, and save results |
| **Real-Time Dashboard** | `realtime_dashboard.py` | Load the trained model, process a live sensor stream, and broadcast anomaly alerts via WebSocket to a web UI |

Both modes use the same **SARN** (Self-Attention Reconstruction Network) model.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                             │
│  NASA C-MAPSS Data → Preprocess → Anomaly Injection → Train SARN   │
│                    → Compute Threshold → Evaluate → Save Results    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                    checkpoints/best_sarn.pt
                    results/pipeline_results.json
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME SYSTEM                                  │
│                                                                     │
│  [Data Source]                                                      │
│       ├── Simulated C-MAPSS (--source sim)                         │
│       ├── REST POST /reading (--source rest)                        │
│       ├── MQTT Broker        (--source mqtt)                        │
│       ├── CSV File Stream    (--source csv)                         │
│       └── WebSocket Client   (--source ws)                          │
│            │                                                        │
│            ▼                                                        │
│  [SlidingWindowBuffer]  ← keeps last 30 timesteps per engine       │
│            │                                                        │
│            ▼                                                        │
│  [SARN Inference Thread] ← anomaly_score = MSE(x, x̂)              │
│            │                                                        │
│            ├──▶ [ThresholdDecision]  score > τ → ANOMALY           │
│            ├──▶ [AlertManager]       severity grading + cooldown   │
│            └──▶ [WebSocket Broadcast] → Live Browser Dashboard      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How the ML Model Works (SARN)

### Core Principle — Unsupervised Reconstruction

SARN is an **autoencoder**. It is trained **only on normal (healthy) data** and learns to perfectly reconstruct it.

- A **normal** window passes through and is reconstructed accurately → **low error**
- An **anomalous** window contains patterns the model has never seen → **high reconstruction error**

The anomaly score for a window is simply the **Mean Squared Error (MSE)** between the input and the reconstruction:

```
score(x) = MSE(x, x̂) = (1 / T·F) · Σ (x - x̂)²
```

### Model Architecture

```
Input (B, T=30, F=17)
  → Linear Projection (F → d_model=64)
  → Learned Positional Encoding
  → Encoder: 3 × Transformer Blocks
      ┌─ LayerNorm
      ├─ Multi-Head Self-Attention: Attn(Q,K,V) = softmax(QKᵀ/√d_k) · V
      └─ Feed-Forward Network (GELU activation)
  → Bottleneck: Linear(64 → 32) → GELU → Linear(32 → 64)
  → Decoder: 3 × Transformer Blocks (same structure)
  → Output Projection (d_model=64 → F=17)
  → x̂ (B, T=30, F=17)
```

### Training Loss

```
L = L_recon + λ · L_reg
L_recon = MSE(x, x̂)            ← reconstruction fidelity
L_reg   = mean(|z|²)           ← L2 regularization on bottleneck latent z
λ       = 1e-4 (default)
```

Optimizer: **AdamW** with Cosine Annealing Warm Restarts scheduler.
Early stopping: patience = 12 epochs.

### Anomaly Detection Threshold

After training on normal data, the threshold `τ` is computed from the reconstruction errors on the validation set:

```
τ = μ_error + k · σ_error     (k = 3.0 by default)
```

At inference time, the optimal threshold is further refined to **maximize F1-score** on the labeled test set.

### Sensor Features Used

The model processes **17 features** per timestep (3 operational settings + 14 sensors):

| Column | Physical Meaning |
|---|---|
| `op1` | Operational setting 1 (altitude/speed regime) |
| `op2` | Operational setting 2 |
| `op3` | Operational setting 3 |
| `s2`  | Total temperature at fan inlet (°R) |
| `s3`  | Total temperature at LPC outlet (°R) |
| `s4`  | Total temperature at HPC outlet (°R) |
| `s7`  | Total pressure at HPC outlet (psia) |
| `s8`  | Physical fan speed (rpm) |
| `s9`  | Physical core speed (rpm) |
| `s11` | Static pressure at HPC outlet (psia) |
| `s12` | Ratio of fuel flow to Ps30 |
| `s13` | Corrected fan speed |
| `s14` | Corrected core speed |
| `s15` | Bypass ratio |
| `s17` | Bleed enthalpy |
| `s20` | High-pressure turbine coolant bleed |
| `s21` | Low-pressure turbine coolant bleed |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full training pipeline
```bash
python run_pipeline.py --subset FD001 --epochs 40
```

### 3. Launch the live dashboard (after training)
```bash
python realtime_dashboard.py
```

### 4. Open your browser
```
http://localhost:5000
```

---

## Running the Full Offline Pipeline

```bash
# Standard run (FD001, 40 epochs)
python run_pipeline.py --subset FD001 --epochs 40

# Quick smoke test (3 epochs, reduced data)
python run_pipeline.py --subset FD001 --epochs 1 --quick

# With LangGraph agent enabled
python run_pipeline.py --subset FD001 --epochs 40 --gemini_key AIzaSy...

# Skip LSTM baseline (faster)
python run_pipeline.py --subset FD001 --epochs 40 --no_lstm

# Skip training entirely and load pre-trained models from checkpoints
python run_pipeline.py --subset FD001 --load_model --gemini_key AIzaSy...

# Custom hyperparameters
python run_pipeline.py --subset FD002 --epochs 60 --d_model 128 --n_heads 8 --d_latent 64
```

### What the Pipeline Does (10 Steps)

| Step | Action |
|---|---|
| 1 | Load & normalize NASA C-MAPSS data |
| 2 | Inject synthetic anomalies into 30% of engines |
| 3 | Build sliding-window sequences (T=30) |
| 4 | Train SARN autoencoder on normal windows only |
| 5 | Train LSTM baseline for comparison |
| 6 | Compute anomaly threshold τ = μ + 3σ |
| 7 | Run inference on test set |
| 8 | Evaluate: F1, AUC-ROC, Precision, Recall, MCC |
| 9 | Run LangGraph agent on top 5 flagged windows |
| 10 | Save all results to `results/` |

---

## Running the Real-Time Live Dashboard

### Source modes

| Command | Description |
|---|---|
| `python realtime_dashboard.py` | Default: internal C-MAPSS simulator |
| `python realtime_dashboard.py --source rest --port 5001` | Wait for external POST requests |
| `python realtime_dashboard.py --source mqtt --broker localhost` | Subscribe to MQTT topic |
| `python realtime_dashboard.py --source csv --file CMAPSSData/test_FD001.txt` | Stream from CSV file |
| `python realtime_dashboard.py --source ws --ws_url ws://host:9000/stream` | Connect to WebSocket source |

Open **http://localhost:5000** to see the live dashboard.

### Dashboard API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Live real-time dashboard UI |
| `/static_dashboard` | GET | Static results dashboard |
| `/reading` | POST | Ingest a sensor reading (REST mode) |
| `/api/summary` | GET | Pipeline training results |
| `/api/live/history` | GET | Last 500 scored windows |
| `/api/live/alerts` | GET | All fired anomaly alerts |
| `/api/live/stats` | GET | Engine processing statistics |
| `/health` | GET | Server health check |

---

## Running the External Telemetry Simulator

The simulator mocks an **FADEC / Ground Station** sending engine telemetry over HTTP.

### Prerequisites
The REST dashboard must already be running:
```bash
# Terminal 1
python realtime_dashboard.py --source rest --port 5001
```

### Start the Simulator
```bash
# Terminal 2
python simulator.py --target http://localhost:5001/reading --engines 2 --hz 5
```

### Inject a Manual Anomaly
While the simulator is running, press:
```
a + ENTER
```
This immediately triggers a violent fault in **Engine #1**. See [anomaly_insertion.md](anomaly_insertion.md) for full details on the injection math.

### Simulator Arguments

| Argument | Default | Description |
|---|---|---|
| `--target` | `http://localhost:5001/reading` | Dashboard REST endpoint |
| `--engines` | `1` | Number of concurrent engines to simulate |
| `--hz` | `5.0` | Sensor readings per second per engine |

---

## All Pipeline Arguments

```
python run_pipeline.py [OPTIONS]

Data:
  --data_dir      CMAPSSData     Path to C-MAPSS folder
  --subset        FD001          Dataset subset: FD001/FD002/FD003/FD004
  --window        30             Sliding window length (timesteps)
  --stride        1              Window extraction stride

Model:
  --d_model       64             Transformer hidden dimension
  --n_heads       4              Number of attention heads
  --n_layers      3              Encoder & decoder depth
  --d_latent      32             Bottleneck latent dimension

Training:
  --epochs        40             Max training epochs
  --batch         64             Batch size
  --lr            3e-4           Learning rate
  --no_lstm                      Skip LSTM baseline training

Anomaly Injection:
  --anomaly_frac  0.30           Fraction of engines to inject anomalies into

Threshold:
  --k_sigma       3.0            τ = μ + k·σ  (multiplier)

Output:
  --out_dir       results        Directory for all results files
  --gemini_key    None           Google Gemini API key for LangGraph agent
  --quick                        Quick test mode (reduced data, 3 epochs min)
  --load_model                   Skip training and load from checkpoints
```

---

## Understanding the Output

After `run_pipeline.py` completes, the `results/` directory contains:

| File | Description |
|---|---|
| `pipeline_results.json` | Full results: metrics, thresholds, training history, agent reports |
| `sarn_scores.npy` | Raw anomaly scores for all test windows |
| `lstm_scores.npy` | LSTM baseline scores for comparison |
| `y_test.npy` | Ground-truth labels (0=normal, 1=anomaly) |
| `comparison.json` | SARN vs LSTM comparative table |

### Example Metrics Output

```
============================================================
  SARN (Ours) — Evaluation Results
============================================================
  Accuracy      : 0.9421  (94.21%)
  Balanced Acc  : 0.9318
  Precision     : 0.8763
  Recall        : 0.9102
  F1 Score      : 0.8929
  MCC           : 0.8712
  AUC-ROC       : 0.9681
  Avg Precision : 0.9214
============================================================
```

### Alert Severity Levels

| Severity | Score Ratio | Recommended Action |
|---|---|---|
| **MEDIUM** | score > τ | Monitor every 5 mins; schedule maintenance within 25 hrs |
| **HIGH** | score > 2τ | Plan precautionary landing; reduce thrust; alert ATC |
| **CRITICAL** | score > 3τ | Declare MAYDAY; shut down engine; divert immediately |

---

## LangGraph Diagnostic Agent

The system includes a **LangGraph multi-step reasoning agent** that:
1. Identifies the top anomalous sensors via per-feature reconstruction error
2. Inspects the attention maps to find which timesteps triggered the alert
3. Queries GPT-4o to generate a **plain-English maintenance report** with fault classification

Enable it with `--gemini_key`:
```bash
python run_pipeline.py --subset FD001 --epochs 40 --gemini_key AIzaSy...
```

Reports are saved in `results/pipeline_results.json` under the `"agent_reports"` key.

---

## Model Checkpoints

Trained models are saved at:
- `checkpoints/best_sarn.pt` — Best SARN model (by validation loss)
- `checkpoints/best_lstm.pt` — Best LSTM baseline

The real-time dashboard auto-loads from these checkpoints on startup.
