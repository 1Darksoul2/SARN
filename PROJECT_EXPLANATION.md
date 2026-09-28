# SARN Aviation Anomaly Detection: Complete Project Guide

This document serves as a comprehensive explanation of your Aviation Anomaly Detection project from a high-level overview down to the mathematical and programmatic minutiae.

---

## 1. Project Overview & Architecture

The goal of this project is to create an intelligent, real-time diagnostic system for aviation engines (turbofans). It monitors live telemetry, detects subtle deviations from normal physical behavior, and automatically generates natural language reports for flight crews using AI.

### The Pipeline Architecture
1. **Data Ingestion & Preprocessing:** Load historical NASA C-MAPSS engine data and normalize it.
2. **Synthetic Anomaly Injection:** Since the raw dataset only has run-to-failure data without labeled intermittent anomalies, we simulate physical engine faults (like spikes, drifts, and noise).
3. **Core Model (SARN):** A Deep Learning Transformer autoencoder trained only on healthy engine data. It detects anomalies by measuring how well it can reconstruct live data.
4. **Agentic Diagnostics (LangGraph + Gemini):** When SARN flags a window, an AI agent takes the anomaly scores and the model's internal "attention maps", and acts as a diagnostic mechanic to identify the root cause.
5. **Real-time Simulator & Dashboard:** A master pipeline that simulates live streaming telemetry from an aircraft to a dashboard interface.

---

## 2. The Data: NASA C-MAPSS

### What is it?
The NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset is the industry standard for predictive maintenance. It simulates the degradation of turbofan engines over time until they fail.

### Preprocessing (`src/data_loader.py`)
- **Raw Features:** It contains 26 columns per timestep (cycle): `engine_id`, `cycle`, 3 operational settings (`op1-op3`), and 21 sensors (`s1-s21`).
- **Feature Selection:** Sensors that do not change value regardless of engine state (e.g., `s1`, `s5` which have zero variance) are dropped. This leaves **17 active features** (3 operational, 14 sensors).
- **Normalization (MinMaxScaler):** All 17 features are scaled mathematically between `0.0` and `1.0`. **This is crucial:** it allows the model to treat pressure metrics (e.g., 500 psia) and temperature metrics (e.g., 1400 °R) with equal mathematical importance.
- **Sliding Windows:** Time-series data is converted into overlapping 3D tensors: `(Batch, Time_Steps, Features)`. By default, it's `(N, 30, 17)`.

---

## 3. Synthetic Anomaly Injection

Because the original NASA dataset lacks labeled, sudden faults (it only has long-term wear-and-tear), `src/anomaly_injector.py` synthetically injects 5 types of anomalies that mimic real-world aviation issues.

1. **Point Anomaly (Sudden Spike):** 
   - *Theory:* A sudden, single-timestep jump.
   - *Math:* $x'(t) = x(t) \pm \alpha \cdot \sigma$
   - *Physical Equivalent:* A bird strike causing a momentary compressor surge or an electrical glitch in a sensor.
2. **Collective Anomaly (Sustained Drift):**
   - *Theory:* A constant offset applied for $L$ timesteps.
   - *Math:* $x'(t+k) = x(t+k) + \beta \cdot \mu$
   - *Physical Equivalent:* A fouled pitot tube or a stuck valve reading.
3. **Trend Anomaly (Progressive Ramp):**
   - *Theory:* A linear degradation over time.
   - *Math:* $x'(t+k) = x(t+k) + (\frac{k}{L-1}) \cdot \gamma \cdot \sigma$
   - *Physical Equivalent:* Gradual loss of oil pressure or developing bearing wear.
4. **Gradual Drift:**
   - *Theory:* A very slow, monotonic drift starting at timestep $t$ and lasting indefinitely.
   - *Math:* $x'(t+k) = x(t+k) + k \cdot \delta$
   - *Physical Equivalent:* Turbine blade erosion over hundreds of flights.
5. **Noise Burst:**
   - *Theory:* High-frequency gaussian noise injection.
   - *Math:* $x'(t+k) = x(t+k) + \mathcal{N}(0, \eta \cdot \sigma)$
   - *Physical Equivalent:* Extreme vibration resonance or electromagnetic interference.

---

## 4. The Model: SARN

**SARN (Self-Attention-Based Reconstruction Network)** in `src/model.py` is the heart of the detection system. 

### Architecture
SARN is a Transformer Autoencoder. Instead of predicting the future, its only job is to recreate the input it was just given.
1. **Input Projection:** The 17 features are mapped to a hidden dimension (`d_model = 64`). Positional encoding is added so the model knows the sequence order.
2. **Transformer Encoder:** Uses Multi-Head Self-Attention. It learns relationships between sensors across time (e.g., "If $Op_1$ is X, $Sensor_4$ should be Y at timestep $t-5$").
3. **Bottleneck:** The data is squeezed into a tiny latent space (`d_latent = 32`). This forces the model to learn the *core physics rules* of the engine, not just memorize data.
4. **Transformer Decoder:** Rebuilds the sequence back to the original 17 features.

### Training Strategy (`src/trainer.py`)
This is an **Unsupervised Anomaly Detector**. 
We train the model **exclusively on healthy, normal engine data**. It becomes an expert at reconstructing a perfectly healthy engine. 
*Loss Function:* $L = MSE(x, \hat{x}) + \lambda_{reg} \cdot mean(|z|^2)$ (Mean Squared Error + Latent Regularization).

### Inference & Global Thresholding
When live data comes in:
1. It passes through SARN.
2. If the data contains an anomaly (e.g., a broken sensor), it breaks the learned physical rules. The bottleneck cannot compress/decompress it properly, leading to a terrible reconstruction.
3. The model calculates a **single global MSE score** for the entire 30x17 window.
4. **The Threshold:** During training, we calculate the Mean ($\mu$) and Std Dev ($\sigma$) of the errors on healthy data. The threshold is set at $\tau = \mu + 3\sigma$. If the live MSE > $\tau$, the alarm triggers.

---

## 5. Multimodal Diagnostic Agent (LangGraph + Gemini)

Once an anomaly is flagged, we need to know *why* it happened. This is handled in `src/agent.py`.

### How it Works:
1. **Perceive:** The Agent receives the raw anomaly window. It looks at the individual reconstruction error for each of the 17 sensors to see which ones failed the hardest.
2. **Attention Maps (The Secret Sauce):** Because SARN uses Self-Attention, we can literally extract the "Attention Maps" (heatmaps) from the model's brain to see *which timesteps* it was focusing on when it got confused.
3. **LLM Chain-of-Thought (Gemini 1.5 Flash):** The agent builds a prompt containing:
   - The global error and severity.
   - The physical names of the top failing sensors (e.g., "Total temperature at HPC outlet").
   - An image (base64) of the Attention Heatmap.
4. **Reason & Report:** Gemini processes this multimodal prompt, acts as an aviation mechanic, and returns a structured Chain-of-Thought diagnosis, including recommendations (e.g., "Immediate diversion to nearest airport").

---

## 6. Real-Time Pipeline & Execution

### `master_runner.py`
This script orchestrates the entire live system using sub-processes:
1. **Pipeline Script (`run_pipeline.py`):** Loads the dataset, injects test anomalies, loads the trained SARN checkpoints, evaluates performance (calculating F1 scores, AUC-ROC), and runs the LangGraph Agent on flagged windows.
2. **Dashboard Server:** Spins up a web server on port 5001 to visualize the live data.
3. **Simulator (`simulator.py`):** Reads the NASA dataset row-by-row and streams it via HTTP POST to the dashboard, mimicking live aircraft telemetry over a satellite link.

### Quick Start
To run the full flow:
```bash
python master_runner.py
```
This single command handles evaluation, spinning up the live web server, and starting the continuous stream of data from the engines.
