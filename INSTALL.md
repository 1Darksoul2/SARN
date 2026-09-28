# Installation Guide — SARN Aviation Anomaly Detection

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| pip | 23+ | `python -m pip install --upgrade pip` |
| CUDA (optional) | 11.8+ | CPU works but is slower for training |
| OpenAI API Key | — | Optional, only for LangGraph diagnostic agent |

---

## Step 1 — Clone / Obtain the Repository

```bash
# If using git:
git clone <your-repo-url>
cd minor

# Or simply navigate to the project folder:
cd d:\minor
```

---

## Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

---

## Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | Version | Role |
|---|---|---|
| `torch` | ≥ 2.0.0 | SARN & LSTM model (PyTorch) |
| `numpy` | ≥ 1.24.0 | Numerical arrays, RNG |
| `pandas` | ≥ 2.0.0 | DataFrame processing |
| `scikit-learn` | ≥ 1.3.0 | Metrics, evaluation |
| `matplotlib` | ≥ 3.7.0 | Static plots |
| `seaborn` | ≥ 0.12.0 | Statistical visualizations |
| `flask` | ≥ 3.0.0 | Web server + REST API |
| `flask-socketio` | ≥ 5.0.0 | WebSocket live broadcast |
| `plotly` | ≥ 5.0.0 | Interactive dashboard charts |
| `langgraph` | ≥ 0.3.0 | Multi-step diagnostic agent |
| `langchain` | ≥ 0.3.0 | LLM tooling |
| `langchain-google-genai` | ≥ 1.0.0 | Gemini connector |

> **Note:** If you have a CUDA GPU, install the CUDA-enabled version of PyTorch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Step 4 — Obtain the NASA C-MAPSS Dataset

The model trains on the **NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS)**.

1. Download from the [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) or [Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).
2. Unzip the files into the `CMAPSSData/` folder in the project root.

The directory should look like:
```
d:\minor\
└── CMAPSSData\
    ├── train_FD001.txt
    ├── test_FD001.txt
    ├── RUL_FD001.txt
    ├── train_FD002.txt
    ├── test_FD002.txt
    ├── RUL_FD002.txt
    ├── train_FD003.txt
    └── ... (etc.)
```

---

## Step 5 — (Optional) Set OpenAI API Key

The LangGraph diagnostic agent uses GPT-4o to explain anomalies in plain English. To enable it:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

Or pass it directly via `--gemini_key AIzaSy...` when running the pipeline.

---

## Step 6 — Verify Installation

```bash
python -c "import torch; import flask; import langgraph; print('All dependencies OK')"
```

Expected output:
```
All dependencies OK
```

---

## Directory Structure After Setup

```
d:\minor\
├── CMAPSSData\          ← NASA dataset files
├── checkpoints\         ← Saved model weights (auto-created)
├── results\             ← Pipeline output JSON + NPY (auto-created)
├── src\
│   ├── anomaly_injector.py   ← 5-type synthetic anomaly injection
│   ├── data_loader.py        ← C-MAPSS parsing & normalization
│   ├── model.py              ← SARN transformer architecture
│   ├── lstm_baseline.py      ← LSTM autoencoder baseline
│   ├── trainer.py            ← Training loop & threshold computation
│   ├── evaluator.py          ← Metrics & comparative analysis
│   ├── realtime_engine.py    ← Threaded inference engine
│   ├── stream_adapters.py    ← MQTT / REST / CSV / WebSocket adapters
│   └── agent.py              ← LangGraph diagnostic agent
├── static\              ← Frontend JS/CSS for dashboard
├── templates\           ← Flask HTML templates
├── run_pipeline.py      ← Main offline training pipeline
├── realtime_dashboard.py ← Live WebSocket dashboard server
├── simulator.py         ← External telemetry simulator
├── dashboard.py         ← Static results dashboard
└── requirements.txt
```
