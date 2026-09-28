"""
Aviation SARN Dashboard — Flask + Plotly
==========================================
Interactive web dashboard for anomaly detection results.
Run:  python dashboard.py
      Open http://localhost:5000
"""

import os
import json
import numpy as np
import flask
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, static_folder="static", template_folder="templates")

RESULTS_DIR  = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "pipeline_results.json")


def load_results():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE) as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/summary")
def api_summary():
    data = load_results()
    if not data:
        return jsonify({"error": "No results found. Run run_pipeline.py first."}), 404
    return jsonify({
        "dataset":     data.get("dataset", "FD001"),
        "n_features":  data.get("n_features", 17),
        "window":      data.get("window", 30),
        "train_shape": data.get("train_shape", []),
        "test_shape":  data.get("test_shape", []),
        "anomaly_rate_train": data.get("anomaly_rate_train", 0),
        "anomaly_rate_test":  data.get("anomaly_rate_test", 0),
        "sarn":  data.get("sarn_result", {}),
        "lstm":  data.get("lstm_result", {}),
        "sarn_threshold": data.get("sarn_threshold", 0),
    })


@app.route("/api/scores")
def api_scores():
    sarn_path  = os.path.join(RESULTS_DIR, "sarn_scores.npy")
    lstm_path  = os.path.join(RESULTS_DIR, "lstm_scores.npy")
    label_path = os.path.join(RESULTS_DIR, "y_test.npy")

    if not os.path.exists(sarn_path):
        return jsonify({"error": "Score files not found."}), 404

    sarn_scores = np.load(sarn_path).tolist()
    labels      = np.load(label_path).tolist() if os.path.exists(label_path) else []
    lstm_scores = np.load(lstm_path).tolist() if os.path.exists(lstm_path) else []

    data = load_results()
    thresh = data.get("sarn_threshold", 0) if data else 0

    return jsonify({
        "sarn_scores": sarn_scores[:2000],   # limit for browser
        "lstm_scores": lstm_scores[:2000],
        "labels":      labels[:2000],
        "threshold":   thresh,
    })


@app.route("/api/history")
def api_history():
    data = load_results()
    if not data:
        return jsonify({"error": "No results."}), 404
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
    data = load_results()
    if not data:
        return jsonify({"error": "No results."}), 404
    return jsonify(data.get("agent_reports", []))


@app.route("/api/comparison")
def api_comparison():
    comp_path = os.path.join(RESULTS_DIR, "comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            return jsonify(json.load(f))
    data = load_results()
    if not data:
        return jsonify({"error": "No results."}), 404
    results = []
    if data.get("sarn_result"):
        results.append(data["sarn_result"])
    if data.get("lstm_result"):
        results.append(data["lstm_result"])
    return jsonify({"results": results})


if __name__ == "__main__":
    print("\n🚀 SARN Aviation Dashboard")
    print("   Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)
