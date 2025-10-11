import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _parse_prob(val):
    # If already numeric
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    # If it's a list/tuple/ndarray like [0.919...]
    if isinstance(val, (list, tuple, np.ndarray)):
        return float(val[0]) if len(val) else np.nan
    # If it's a string, maybe like "[0.9190303087234497]" or "0.9190"
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return float(s) if s else np.nan

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize column names just in case
    # df = df.rename(columns={c: c.lower() for c in df.columns})
    # print(df.columns)
    # Require 'Target' and 'prob_fake'
    if not {'Label', 'prob_fake'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'labe;' and 'prob_fake'.")

    y_true = df['Label'].astype(int).to_numpy()
    # <<< Only change: robustly parse prob_fake to remove [] like "[0.919...]"
    # y_score = df['prob_fake'].apply(_parse_prob).astype(float).to_numpy()
    y_score = df['prob_fake'].apply(_parse_prob).astype(float).to_numpy() 


    # Predict label: threshold 0.5 on prob_fake
    y_pred = (y_score >= 0.5).astype(int)

    # Confusion terms using your mapping
    TP = np.sum((y_pred == 0) & (y_true == 0))
    TN = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 0) & (y_true == 1))
    FN = np.sum((y_pred == 1) & (y_true == 0))

    # Metrics
    acc = np.mean(y_pred == y_true)

    # AUC (guard when only one class is present)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")  # Undefined when only one class present

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return {
        "ACC": acc,
        "AUC": auc,
        "FAR": FAR,
        "FRR": FRR,
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "N": int(len(df))
    }

def main():
    ap = argparse.ArgumentParser(description="Compute ACC, AUC, FAR, FRR from label/prob_fake CSV.")
    ap.add_argument("--input_csv", default='/mnt/ssd/project/lilin/politic_deepfakes/AI-Face-FairnessBench-main/Forensics-Bench/vlm_processed/random_pick_10_monkey-chat.csv', help="Path to CSV with columns: label, prob_fake")
    args = ap.parse_args()

    metrics = compute_metrics(args.input_csv)

    # Pretty print (rounded to 4 decimals where applicable)
    print(f"N   : {metrics['N']}")
    print(f"TP  : {metrics['TP']}  TN: {metrics['TN']}  FP: {metrics['FP']}  FN: {metrics['FN']}")
    print(f"ACC : {metrics['ACC']:.4f}")
    print(f"AUC : {metrics['AUC']:.4f}" if np.isfinite(metrics['AUC']) else "AUC : NaN (only one class present)")
    print(f"FAR : {metrics['FAR']:.4f}")
    print(f"FRR : {metrics['FRR']:.4f}")

if __name__ == "__main__":
    main()
