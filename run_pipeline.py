"""
Main Pipeline — SARN Aviation Anomaly Detection
================================================
Runs the complete pipeline:
  1. Load NASA C-MAPSS dataset
  2. Inject synthetic anomalies (5 types)
  3. Train SARN + LSTM Baseline
  4. Evaluate & Compare
  5. Run LangGraph diagnostic agent on flagged windows
  6. Save results for dashboard

Usage:
    python run_pipeline.py --subset FD001 --epochs 40
    python run_pipeline.py --subset FD001 --epochs 1 --quick  (quick test)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader    import build_datasets, make_sequences, SENSOR_COLS, FEATURE_COLS
from src.anomaly_injector import inject_anomalies
from src.model          import build_sarn
from src.lstm_baseline  import build_lstm
from src.trainer        import prepare_normal_loader, train_model, compute_threshold, load_checkpoint
from src.evaluator      import evaluate_detector, compare_models, find_optimal_threshold
from src.agent          import run_diagnostic_agent


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="SARN Aviation Anomaly Detection Pipeline")
    p.add_argument("--data_dir",  default="CMAPSSData",      help="Path to CMAPSS data folder")
    p.add_argument("--subset",    default="FD001",           help="FD001 / FD002 / FD003 / FD004")
    p.add_argument("--window",    type=int, default=30,      help="Sliding window length")
    p.add_argument("--stride",    type=int, default=1,       help="Window stride")
    p.add_argument("--epochs",    type=int, default=40,      help="Training epochs")
    p.add_argument("--batch",     type=int, default=64,      help="Batch size")
    p.add_argument("--lr",        type=float, default=3e-4,  help="Learning rate")
    p.add_argument("--d_model",   type=int, default=64,      help="SARN d_model")
    p.add_argument("--n_heads",   type=int, default=4,       help="Attention heads")
    p.add_argument("--n_layers",  type=int, default=3,       help="Encoder/Decoder layers")
    p.add_argument("--d_latent",  type=int, default=32,      help="Bottleneck dim")
    p.add_argument("--anomaly_frac", type=float, default=0.30, help="Fraction of engines to inject anomalies")
    p.add_argument("--k_sigma",   type=float, default=3.0,   help="Threshold sigma multiplier")
    p.add_argument("--no_lstm",   action="store_true",       help="Skip LSTM baseline")
    p.add_argument("--load_model", action="store_true",      help="Skip training and load from checkpoints")
    p.add_argument("--quick",     action="store_true",       help="Quick test (1 epoch, small data)")
    p.add_argument("--gemini_key", default=None,             help="Google Gemini API key for LLM diagnosis")
    p.add_argument("--out_dir",   default="results",         help="Results output directory")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  SARN Aviation Anomaly Detection Pipeline")
    print(f"  Dataset: NASA C-MAPSS {args.subset}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("── Step 1: Loading & Preprocessing ──")
    tr_df, te_df, scaler = build_datasets(
        data_dir=args.data_dir,
        subset=args.subset,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch,
    )
    print(f"  Train engines: {tr_df['engine_id'].nunique()}")
    print(f"  Test  engines: {te_df['engine_id'].nunique()}")

    # ── 2. Inject anomalies ───────────────────────────────────────────────────
    print("\n── Step 2: Synthetic Anomaly Injection ──")
    tr_df_inj = inject_anomalies(tr_df, SENSOR_COLS,
                                  anomaly_fraction=args.anomaly_frac,
                                  region="early", seed=42)
    te_df_inj = inject_anomalies(te_df, SENSOR_COLS,
                                  anomaly_fraction=args.anomaly_frac,
                                  region="late", seed=123)

    # ── 3. Build sequences ────────────────────────────────────────────────────
    print("\n── Step 3: Building Sequences ──")
    X_train, y_train, eids_train = make_sequences(tr_df_inj, args.window, args.stride, FEATURE_COLS)
    X_test,  y_test,  eids_test  = make_sequences(te_df_inj, args.window, 1,           FEATURE_COLS)

    print(f"  Train: {X_train.shape}  Anomaly rate: {y_train.mean():.2%}")
    print(f"  Test : {X_test.shape}   Anomaly rate: {y_test.mean():.2%}")

    n_features = X_train.shape[2]

    if args.quick:
        print("  [QUICK MODE] Reducing dataset size")
        X_train, y_train = X_train[:2000], y_train[:2000]
        X_test,  y_test  = X_test[:500],   y_test[:500]
        args.epochs = max(args.epochs, 3)

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    tr_loader, val_loader = prepare_normal_loader(
        X_train, y_train, batch_size=args.batch
    )

    # ── 5. Train SARN ─────────────────────────────────────────────────────────
    print(f"\n── Step 4: SARN Model Setup ──")
    sarn = build_sarn(
        n_features=n_features,
        window=args.window,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        d_latent=args.d_latent,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
    )
    
    sarn_history = None
    sarn_ckpt = os.path.join("checkpoints", "best_sarn.pt")
    if args.load_model and os.path.exists(sarn_ckpt):
        print(f"  [INFO] Loading SARN from {sarn_ckpt}")
        sarn = load_checkpoint(sarn, sarn_ckpt, device=device)
    else:
        print(f"  [INFO] Training SARN...")
        sarn_history, sarn = train_model(
            sarn, tr_loader, val_loader,
            model_type="sarn",
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_dir="checkpoints",
        )
        with open(os.path.join("checkpoints", "sarn_history.json"), "w") as f:
            json.dump(sarn_history, f)
        
    if args.load_model and sarn_history is None:
        hist_path = os.path.join("checkpoints", "sarn_history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                sarn_history = json.load(f)

    # ── 6. Train LSTM baseline ────────────────────────────────────────────────
    lstm_history = None
    lstm = None
    if not args.no_lstm:
        print(f"\n── Step 5: LSTM Baseline Setup ──")
        lstm = build_lstm(n_features=n_features,
                          hidden_size=args.d_model,
                          n_layers=2, dropout=0.1)
        
        lstm_ckpt = os.path.join("checkpoints", "best_lstm.pt")
        if args.load_model and os.path.exists(lstm_ckpt):
            print(f"  [INFO] Loading LSTM from {lstm_ckpt}")
            lstm = load_checkpoint(lstm, lstm_ckpt, device=device)
        else:
            print(f"  [INFO] Training LSTM...")
            lstm_history, lstm = train_model(
                lstm, tr_loader, val_loader,
                model_type="lstm",
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                checkpoint_dir="checkpoints",
            )
            with open(os.path.join("checkpoints", "lstm_history.json"), "w") as f:
                json.dump(lstm_history, f)
                
        if args.load_model and lstm_history is None:
            hist_path = os.path.join("checkpoints", "lstm_history.json")
            if os.path.exists(hist_path):
                with open(hist_path, "r") as f:
                    lstm_history = json.load(f)

    # ── 7. Thresholds ─────────────────────────────────────────────────────────
    print(f"\n── Step 6: Computing Thresholds ──")
    X_normal = X_train[y_train == 0]

    sarn.to(device)
    sarn_thresh, sarn_mu, sarn_sig = compute_threshold(
        sarn, X_normal, k=args.k_sigma, device=device
    )
    if lstm is not None:
        lstm.to(device)
        lstm_thresh, lstm_mu, lstm_sig = compute_threshold(
            lstm, X_normal, k=args.k_sigma, device=device
        )

    # ── 8. Inference ──────────────────────────────────────────────────────────
    print(f"\n── Step 7: Running Inference on Test Set ──")
    sarn.eval()
    sarn_scores = []
    X_t = torch.tensor(X_test, dtype=torch.float32)
    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH):
            b = X_t[i:i+BATCH].to(device)
            sarn_scores.append(sarn.anomaly_score(b).cpu().numpy())
    sarn_scores = np.concatenate(sarn_scores)

    lstm_scores = None
    if lstm is not None:
        lstm.eval()
        lstm_sc = []
        with torch.no_grad():
            for i in range(0, len(X_t), BATCH):
                b = X_t[i:i+BATCH].to(device)
                lstm_sc.append(lstm.anomaly_score(b).cpu().numpy())
        lstm_scores = np.concatenate(lstm_sc)

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n── Step 8: Evaluation ──")
    results = []

    # Optional: find optimal threshold
    if len(np.unique(y_test)) > 1:
        opt_thresh_sarn = find_optimal_threshold(sarn_scores, y_test, method="f1")
        print(f"[SARN] Optimal F1 threshold: {opt_thresh_sarn:.6f} (was {sarn_thresh:.6f})")
        sarn_thresh_eval = opt_thresh_sarn
    else:
        sarn_thresh_eval = sarn_thresh

    sarn_result = evaluate_detector(sarn_scores, y_test, sarn_thresh_eval, "SARN (Ours)")
    results.append(sarn_result)

    if lstm_scores is not None and len(np.unique(y_test)) > 1:
        opt_thresh_lstm = find_optimal_threshold(lstm_scores, y_test, method="f1")
        lstm_thresh_eval = opt_thresh_lstm
        lstm_result = evaluate_detector(lstm_scores, y_test, lstm_thresh_eval, "LSTM Baseline")
        results.append(lstm_result)

    if len(results) > 1:
        winner = compare_models(results, save_dir=args.out_dir)

    # ── 10. LangGraph Agent on flagged windows ────────────────────────────────
    print(f"\n── Step 9: LangGraph Diagnostic Agent ──")
    flagged_idxs = np.where(sarn_scores >= sarn_thresh_eval)[0]
    print(f"  Flagged {len(flagged_idxs)} anomalous windows from test set")

    agent_reports = []
    n_agent = min(5, len(flagged_idxs))
    for i in range(n_agent):
        idx    = flagged_idxs[i]
        window = X_test[idx]
        eid    = eids_test[idx] if idx < len(eids_test) else 0
        report = run_diagnostic_agent(
            window=window,
            feature_names=FEATURE_COLS,
            model=sarn,
            threshold=sarn_thresh_eval,
            engine_id=int(eid),
            cycle=i,
            gemini_api_key=args.gemini_key,
        )
        agent_reports.append(report)

    # ── 11. Save all results ──────────────────────────────────────────────────
    print(f"\n── Step 10: Saving Results ──")

    # Training histories
    np.save(os.path.join(args.out_dir, "sarn_scores.npy"), sarn_scores)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)
    if lstm_scores is not None:
        np.save(os.path.join(args.out_dir, "lstm_scores.npy"), lstm_scores)

    pipeline_results = {
        "dataset":     args.subset,
        "window":      args.window,
        "n_features":  n_features,
        "sarn_threshold": sarn_thresh_eval,
        "sarn_result": sarn_result,
        "lstm_result": results[1] if len(results) > 1 else None,
        "agent_reports": agent_reports,
        "sarn_history": sarn_history,
        "lstm_history": lstm_history,
        "train_shape": list(X_train.shape),
        "test_shape":  list(X_test.shape),
        "anomaly_rate_train": float(y_train.mean()),
        "anomaly_rate_test":  float(y_test.mean()),
    }

    with open(os.path.join(args.out_dir, "pipeline_results.json"), "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    print(f"\n  ✅ Results saved to '{args.out_dir}/'")
    print(f"  📊 Run dashboard: python dashboard.py")
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"  SARN  → F1={sarn_result['f1']:.4f}  AUC-ROC={sarn_result['auc_roc']:.4f}")
    if len(results) > 1:
        print(f"  LSTM  → F1={results[1]['f1']:.4f}  AUC-ROC={results[1]['auc_roc']:.4f}")
    print(f"{'='*60}\n")

    return pipeline_results


if __name__ == "__main__":
    main()
