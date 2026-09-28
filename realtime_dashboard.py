"""
SARN Live Dashboard — Real-Time WebSocket Server
=================================================
Combines the static results dashboard with a live inference feed.

Run:
    python realtime_dashboard.py              ← uses C-MAPSS simulator
    python realtime_dashboard.py --source rest  ← waits for POST /reading
    python realtime_dashboard.py --source mqtt --broker localhost
    python realtime_dashboard.py --source csv --file CMAPSSData/test_FD001.txt

Open: http://localhost:5000
"""

import os
import sys
import json
import time
import argparse
import threading
import numpy as np

import flask
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.realtime_engine import RealtimeInferenceEngine, Alert
from src.data_loader import FEATURE_COLS


# ── Flask + SocketIO app ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "sarn-aviation-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

RESULTS_FILE = "results/pipeline_results.json"

# ── Global inference engine (set in main) ────────────────────────────────────
_engine: RealtimeInferenceEngine = None

# ── In-memory live data store (last 500 readings) ────────────────────────────
_live_history = []
_live_alerts  = []
_MAX_HISTORY  = 500
_lock = threading.Lock()


# ── Callbacks from inference engine ──────────────────────────────────────────
def on_result(result: dict):
    """Called every time a window is scored. Broadcasts to all connected clients."""
    with _lock:
        _live_history.append(result)
        if len(_live_history) > _MAX_HISTORY:
            _live_history.pop(0)

    socketio.emit("live_result", result)


def on_alert(alert: Alert):
    """Called only on anomaly alerts. Broadcasts high-priority event."""
    alert_dict = {
        "engine_id":    alert.engine_id,
        "cycle":        alert.cycle,
        "timestamp":    alert.timestamp,
        "score":        alert.score,
        "threshold":    alert.threshold,
        "severity":     alert.severity,
        "top_sensor":   alert.top_sensor,
        "message":      alert.message,
        "recommendations": alert.recommendations,
    }
    with _lock:
        _live_alerts.append(alert_dict)
        if len(_live_alerts) > 100:
            _live_alerts.pop(0)

    socketio.emit("live_alert", alert_dict)
    print(f"[Dashboard] Alert broadcast: {alert.severity} Engine #{alert.engine_id}")


# ── HTTP routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("realtime.html")


@app.route("/static_dashboard")
def static_dashboard():
    return render_template("dashboard.html")


@app.route("/api/summary")
def api_summary():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results found. Run run_pipeline.py first."}), 404
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return jsonify({
        "dataset":     data.get("dataset"),
        "n_features":  data.get("n_features"),
        "window":      data.get("window"),
        "train_shape": data.get("train_shape"),
        "test_shape":  data.get("test_shape"),
        "anomaly_rate_train": data.get("anomaly_rate_train"),
        "anomaly_rate_test":  data.get("anomaly_rate_test"),
        "sarn":  data.get("sarn_result"),
        "lstm":  data.get("lstm_result"),
        "sarn_threshold": data.get("sarn_threshold"),
    })


@app.route("/api/scores")
def api_scores():
    RESULTS_DIR = "results"
    sarn_path  = os.path.join(RESULTS_DIR, "sarn_scores.npy")
    lstm_path  = os.path.join(RESULTS_DIR, "lstm_scores.npy")
    label_path = os.path.join(RESULTS_DIR, "y_test.npy")

    if not os.path.exists(sarn_path):
        return jsonify({"error": "Score files not found."}), 404

    sarn_scores = np.load(sarn_path).tolist()
    labels      = np.load(label_path).tolist() if os.path.exists(label_path) else []
    lstm_scores = np.load(lstm_path).tolist() if os.path.exists(lstm_path) else []

    thresh = 0
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
            thresh = data.get("sarn_threshold", 0)

    return jsonify({
        "sarn_scores": sarn_scores[:2000],
        "lstm_scores": lstm_scores[:2000],
        "labels":      labels[:2000],
        "threshold":   thresh,
    })


@app.route("/api/history")
def api_history():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results."}), 404
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    sh = data.get("sarn_history", {})
    lh = data.get("lstm_history", {})
    return jsonify({
        "sarn_train": sh.get("train_loss", []) if sh else [],
        "sarn_val":   sh.get("val_loss", []) if sh else [],
        "lstm_train": lh.get("train_loss", []) if lh else [],
        "lstm_val":   lh.get("val_loss", []) if lh else [],
    })


@app.route("/api/agent_reports")
def api_agent_reports():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results."}), 404
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return jsonify(data.get("agent_reports", []))


@app.route("/api/comparison")
def api_comparison():
    RESULTS_DIR = "results"
    comp_path = os.path.join(RESULTS_DIR, "comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            return jsonify(json.load(f))
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results."}), 404
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    results = []
    if data.get("sarn_result"):
        results.append(data["sarn_result"])
    if data.get("lstm_result"):
        results.append(data["lstm_result"])
    return jsonify({"results": results})


@app.route("/api/live/history")
def api_live_history():
    with _lock:
        return jsonify(list(_live_history))


@app.route("/api/live/alerts")
def api_live_alerts():
    with _lock:
        return jsonify(list(_live_alerts))


@app.route("/api/live/stats")
def api_live_stats():
    if _engine:
        return jsonify(_engine.stats)
    return jsonify({})


# ── REST ingestion endpoint (when using RestApiAdapter) ──────────────────────
@app.route("/reading", methods=["GET", "POST"])
def ingest_reading():
    """External systems POST sensor readings here."""
    if request.method == "GET":
        return jsonify({
            "message": "POST JSON telemetry to this endpoint.",
            "required_fields": ["engine_id", "cycle", "features", "timestamp"],
        })

    data  = request.get_json(force=True)
    eid   = int(data.get("engine_id", 1))
    cycle = int(data.get("cycle", 0))
    feats = data.get("features", {})
    ts    = float(data.get("timestamp", time.time()))

    if _engine:
        _engine.push_dict(eid, cycle, feats, timestamp=ts)
        return jsonify({"status": "ok", "queued": True})
    return jsonify({"status": "error", "message": "Engine not started"}), 503


@app.route("/health")
def health():
    return jsonify({
        "status":  "running",
        "engine":  _engine.stats if _engine else None,
        "history": len(_live_history),
        "alerts":  len(_live_alerts),
    })


# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    with _lock:
        emit("history_snapshot", {
            "history": list(_live_history[-100:]),
            "alerts":  list(_live_alerts[-20:]),
        })
    print(f"[Dashboard] Client connected — sent {min(len(_live_history),100)} history points")


# ── Main entry point ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SARN Real-Time Dashboard")
    p.add_argument("--source",   default="sim",
                   choices=["sim","rest","mqtt","csv","ws"],
                   help="Data source adapter")
    p.add_argument("--checkpoint", default="checkpoints/best_sarn.pt")
    p.add_argument("--threshold",  type=float, default=None)
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--port",       type=int,   default=5000)
    p.add_argument("--delay",      type=float, default=0.15,
                   help="Seconds between readings for CSV/sim adapters")
    # MQTT options
    p.add_argument("--broker",  default="localhost")
    p.add_argument("--topic",   default="aircraft/engine/readings")
    # CSV option
    p.add_argument("--file",    default="CMAPSSData/test_FD001.txt")
    # WebSocket option
    p.add_argument("--ws_url",  default="ws://localhost:9000/stream")
    return p.parse_args()


def main():
    global _engine
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  SARN Real-Time Aviation Anomaly Detection")
    print(f"  Source: {args.source.upper()}")
    print(f"{'='*55}\n")

    # Build inference engine
    _engine = RealtimeInferenceEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device="cpu",    # use cpu for real-time (avoids CUDA latency spikes)
        cooldown_cycles=5,
    )
    _engine.on_result(on_result)
    _engine.on_alert(on_alert)
    _engine.start()

    # Start data source adapter
    if args.source == "sim":
        from src.stream_adapters import SimulatedCMAPSSAdapter
        adapter = SimulatedCMAPSSAdapter(
            _engine, data_dir="CMAPSSData", subset="FD001",
            delay_s=args.delay, inject_anomaly=True
        )
        adapter.start()

    elif args.source == "rest":
        print(f"[Main] REST mode — POST to http://localhost:{args.port}/reading")

    elif args.source == "mqtt":
        from src.stream_adapters import MqttAdapter
        adapter = MqttAdapter(_engine, broker=args.broker, topic=args.topic)
        adapter.start()

    elif args.source == "csv":
        from src.stream_adapters import CsvStreamAdapter
        adapter = CsvStreamAdapter(
            filepath=args.file, engine=_engine,
            engine_id=1, delay_s=args.delay, loop=True
        )
        adapter.start()

    elif args.source == "ws":
        from src.stream_adapters import WebSocketClientAdapter
        adapter = WebSocketClientAdapter(_engine, ws_url=args.ws_url)
        adapter.start()

    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  Static results: http://localhost:{args.port}/static_dashboard")
    print(f"  Health check:   http://localhost:{args.port}/health\n")

    socketio.run(app, host=args.host, port=args.port,
                 debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
