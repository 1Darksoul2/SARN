# Anomaly Insertion — Complete Technical Reference

> **How synthetic faults are mathematically created and injected into NASA C-MAPSS sensor data for model training and evaluation.**

---

## Why Inject Anomalies?

NASA C-MAPSS contains **no labeled anomaly events** — only continuous run-to-failure degradation data. To train and properly evaluate a binary anomaly detector, we synthetically generate **five physics-inspired anomaly types** and inject them at specific points in the engine timelines. Each injected timestep is labeled `anomaly = 1` in the dataset; all other timesteps are `anomaly = 0`.

---

## What Data is Being Modified

Every sensor value in the dataset is **min-max normalized to `[0.0, 1.0]`** before injection. The 17 features available for injection are:

| Column | Physical Sensor |
|---|---|
| `op1`, `op2`, `op3` | Operational condition settings |
| `s2` | Total temperature at fan inlet (°R) |
| `s3` | Total temperature at LPC outlet (°R) |
| `s4` | Total temperature at HPC outlet (°R) |
| `s7` | Total pressure at HPC outlet (psia) |
| `s8` | Physical fan speed (rpm) |
| `s9` | Physical core speed (rpm) |
| `s11` | Static pressure at HPC outlet (psia) |
| `s12` | Fuel flow ratio (Wf/Ps30) |
| `s13` | Corrected fan speed |
| `s14` | Corrected core speed |
| `s15` | Bypass ratio |
| `s17` | Bleed enthalpy |
| `s20` | HP turbine coolant bleed |
| `s21` | LP turbine coolant bleed |

> **Clipping Rule:** All injected values are always clipped to `[0.0, 1.0]` to stay within the normalized range. This prevents physically impossible values.

---

## Injection Strategy

Anomalies are injected per-engine, not per-sensor. For each chosen engine:
1. **One anomaly type** is randomly selected from the five types below.
2. **One or more sensors** from the full feature list are randomly selected to perturb.
3. The injection window is placed in a **region** of the engine's lifecycle based on whether it's training or test data.

### Engine Selection

```python
n_anomalous = max(1, int(n_engines × anomaly_fraction))
# Default anomaly_fraction = 0.30
# → 30% of all engines receive an anomaly injection
```

### Time Window Region

| Dataset Split | Injection Region | Rationale |
|---|---|---|
| **Training set** | First **70%** of engine life (`region='early'`) | Model sees anomalies early in the lifecycle, not during end-of-life degradation |
| **Test set** | Last **30%** of engine life (`region='late'`) | Anomalies appear pre-failure, simulating real-world fault onset |

```python
# region = 'early': injection start t ∈ [0, 0.70·N - L]
# region = 'late':  injection start t ∈ [0.70·N, N - L]
# N = total cycles for this engine, L = anomaly window length
```

---

## The Five Anomaly Types

For all formulas below:
- `x(t, s)` = original normalized sensor value at timestep `t` for sensor `s`
- `x'(t, s)` = perturbed (anomalous) value
- `μ_s` = global mean of sensor `s` across the entire dataset
- `σ_s` = global standard deviation of sensor `s` (floored at `1e-6`)
- All random variables are drawn fresh per injection event

---

### Type 1 — Point Anomaly (Sudden Spike / Drop)

**Formula:**
```
x'(t, s) = clip( x(t, s) ± α · σ_s,  0.0, 1.0 )

α ~ Uniform(3.0, 6.0)
direction ∈ {-1, +1}   (chosen randomly)
Affects: 1–3 sensors simultaneously, 1 timestep only
```

**What it simulates:**  
An instantaneous, high-magnitude deviation — like a **bird-strike event**, a **compressor surge blip**, or a sudden EGT spike caused by a transient fuel delivery fault.

**Code:**
```python
alpha     = RNG.uniform(3.0, 6.0)
direction = RNG.choice([-1, 1])
df.loc[t, s] = np.clip(df.iloc[t][s] + direction * alpha * sig[s], 0.0, 1.0)
```

**Example (σ_s = 0.04):**  
- `α = 4.5`, `direction = +1` → sensor jumps by `+4.5 × 0.04 = +0.18` in one cycle.  
- If baseline was 0.35, it spikes to `0.53` instantly.

---

### Type 2 — Collective / Contextual Anomaly (Sustained Drift)

**Formula:**
```
x'(t+k, s) = clip( x(t+k, s) + β · μ_s,  0.0, 1.0 )    for k = 0, 1, ..., L-1

β ~ Uniform(0.20, 0.45)
L ~ Uniform(5, 15)   (injection window length, in cycles)
Affects: 1–2 sensors simultaneously
```

**What it simulates:**  
A sensor that is **biased or stuck** — reading consistently higher than the true value. Physically represents a **fouled pitot tube**, a **clogged fuel filter** causing sustained over-reading, or a **sensor calibration offset**.

**Code:**
```python
beta = RNG.uniform(0.20, 0.45)
L    = RNG.integers(5, 16)   # [5, 15]
idxs = df.index[t: t + L]
df.loc[idxs, s] = np.clip(df.loc[idxs, s] + beta * mu[s], 0.0, 1.0)
```

**Example (μ_s = 0.5):**  
- `β = 0.30`, `L = 10` → sensor is inflated by `0.30 × 0.50 = +0.15` for 10 consecutive cycles.

---

### Type 3 — Trend / Ramp Anomaly (Progressive Fault Onset)

**Formula:**
```
x'(t+k, s) = clip( x(t+k, s) + (k / (L-1)) · γ · σ_s,  0.0, 1.0 )    for k = 0, 1, ..., L-1

γ ~ Uniform(2.0, 4.0)
L ~ Uniform(8, 20)   (ramp length in cycles)
Affects: 1–2 sensors simultaneously
```

The ramp factor `k / (L-1)` linearly grows from `0.0` at the start to `1.0` at the end of the window, creating a smooth, escalating deviation.

**What it simulates:**  
A **gradually worsening fault** — such as a developing bearing fatigue crack, **progressive oil loss raising EGT**, or **turbine blade erosion** that increasingly stresses connected sensors.

**Code:**
```python
gamma     = RNG.uniform(2.0, 4.0)
L         = RNG.integers(8, 21)   # [8, 20]
actual_L  = len(idxs)
ramp      = np.linspace(0, 1, actual_L)    # 0 → 1 linearly
df.loc[idxs, s] = np.clip(df.loc[idxs, s] + ramp * gamma * sig[s], 0.0, 1.0)
```

**Example (σ_s = 0.04):**  
- `γ = 3.0`, `L = 10` → sensor drifts from `+0.00` at cycle 0 to `+3.0 × 0.04 = +0.12` by cycle 9.

---

### Type 4 — Gradual Drift (Slow Monotonic Degradation)

**Formula:**
```
x'(t+k, s) = clip( x(t+k, s) + k · δ,  0.0, 1.0 )    for k = 0, 1, ..., N-start-1

δ ~ Uniform(0.002, 0.008)
Starts at random cycle, continues to end of engine life (no fixed window)
Affects: 1–2 sensors simultaneously
```

Unlike the trend anomaly, this one does **not stop** — it drifts indefinitely from the injection start to the last cycle of the engine's data, producing a slow, persistent creep.

**What it simulates:**  
**Long-term slow degradation** of a specific component — like gradual turbine blade erosion, a slow sensor drift due to thermal cycling fatigue, or progressive oil leak causing small but cumulative temperature rise.

**Code:**
```python
delta       = RNG.uniform(0.002, 0.008)
L           = n - start          # runs to end of engine data
drift_steps = np.arange(L)       # [0, 1, 2, ..., L-1]
df.loc[idxs, s] = np.clip(df.loc[idxs, s] + drift_steps * delta, 0.0, 1.0)
```

**Example:**  
- `δ = 0.005` → sensor increases by `0.005` per cycle. After 50 cycles: `+0.25` total cumulative drift.

---

### Type 5 — Noise Burst (High-Frequency Interference)

**Formula:**
```
x'(t+k, s) = clip( x(t+k, s) + ε_k,  0.0, 1.0 )    for k = 0, 1, ..., L-1

ε_k ~ Normal(0, η · σ_s)   (independent sample each timestep)
η ~ Uniform(2.0, 4.0)
L ~ Uniform(5, 15)
Affects: 1–2 sensors simultaneously
```

Each timestep within the window receives an **independent random perturbation** from a zero-mean Gaussian scaled by `η × σ_s`. Unlike the other types, this noise can go both positive and negative within the same injection window.

**What it simulates:**  
**Electromagnetic interference (EMI)** corrupting a sensor's signal, **vibration resonance** causing erratic readings, or **electrical noise** from nearby avionics disrupting the sensor bus.

**Code:**
```python
eta   = RNG.uniform(2.0, 4.0)
L     = RNG.integers(5, 16)
noise = RNG.normal(0, eta * sig[s], actual_L)   # independent per timestep
df.loc[idxs, s] = np.clip(df.loc[idxs, s] + noise, 0.0, 1.0)
```

**Example (σ_s = 0.04):**  
- `η = 3.0` → noise std = `3.0 × 0.04 = 0.12`. Each cycle gets ±0.12 random jitter for 5–15 cycles.

---

## Summary Comparison Table

| Type | Formula | Duration | Physics Model | Severity |
|---|---|---|---|---|
| **Point** | `x ± α·σ` | 1 timestep | Bird-strike, compressor surge | Very high impulse |
| **Collective** | `x + β·μ` | 5–15 cycles | Fouled sensor, calibration offset | Moderate, sustained |
| **Trend** | `x + (k/L-1)·γ·σ` | 8–20 cycles | Developing bearing fault, oil loss | Low→High escalation |
| **Drift** | `x + k·δ` | Until end | Blade erosion, long-term sensor aging | Very slow, cumulative |
| **Noise Burst** | `x + N(0, η·σ)` | 5–15 cycles | EMI, vibration resonance | Erratic, random |

---

## Live Simulator Anomaly Injection (simulator.py)

The **external telemetry simulator** (`simulator.py`) has a simpler, real-time anomaly injection mechanism compared to the offline injector above. It is designed for quick manual testing of the live dashboard.

### Normal State — Baseline Wear & Tear

Every cycle, all 17 sensors are updated with a physics-inspired slow degradation formula:

```
state[s] = clip( state[s] + direction_s × degradation_factor + ε,  0.0, 1.0 )

ε                 ~ Normal(0, 0.005)             per sensor, per cycle
degradation_factor ~ Uniform(0.0001, 0.0005)     fixed per engine at startup
direction_s       = +1 for s3, s4, s8, s11, s13, s15, s17   (rise with age)
                  = -1 for all other sensors                  (fall with age)
```

This models the physical reality that some parameters (e.g., exhaust temperatures) increase as an engine ages, while others (e.g., pressures, speeds) decrease.

### Manual Anomaly Trigger (Press `a + ENTER`)

Triggering an anomaly immediately sets:

```python
anomaly_active  = True
anomaly_sensor  = random.choice(["s2", "s3", "s4", "s9", "s14"])
anomaly_severity = random.choice([+0.15, -0.15, +0.25, -0.25])
```

The targeted sensor pool (`s2`, `s3`, `s4`, `s9`, `s14`) represents the most critical engine health indicators (fan inlet temperature, LPC temperature, HPC temperature, core speed, corrected core speed).

From this point forward, **every cycle** applies the severity to the selected sensor:

```
state[anomaly_sensor] = clip( state[anomaly_sensor] + anomaly_severity,  0.0, 1.0 )

anomaly_severity ∈ {+0.15, -0.15, +0.25, -0.25}   (magnitude: 0.15 or 0.25)
```

**Why this is so visible:** The severity magnitude (0.15–0.25 per cycle) is **30× to 2500× larger** than normal degradation (0.0001–0.0005 per cycle). The sensor rapidly saturates to its floor (`0.0`) or ceiling (`1.0`) within a few cycles, mimicking a catastrophic, acute failure.

---

## Labeling

Every timestep modified by any of the 5 offline injection types is marked:

```python
df.loc[injected_indices, "anomaly"] = 1   # 1 = anomaly
# All other timesteps remain:
df["anomaly"] = 0                         # 0 = normal
```

This label is used as **ground truth** during evaluation (F1, AUC-ROC, etc.) but is **never seen by the SARN model during training** — it only trains on normal windows.

---

## Reproducibility

All injection RNG operations use a seeded NumPy Generator:

```python
RNG = np.random.default_rng(seed=42)   # training set injection
RNG = np.random.default_rng(seed=123)  # test set injection
```

This ensures fully reproducible anomaly placement across runs.
