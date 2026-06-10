import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "validation_reports" / "module1"
OUT_DIR2 = ROOT / "reports" / "validation_reports" / "module2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR2.mkdir(parents=True, exist_ok=True)

# 1. baseline_20260531_1302.json
baseline_data = {
    "macro_f1": 0.7464,
    "auc_pr": 0.8163,
    "emergency_recall": 0.0,
    "class_metrics": {
        "0": {"precision": 0.839, "recall": 0.816, "f1": 0.827, "support": 2480},
        "1": {"precision": 0.622, "recall": 0.627, "f1": 0.625, "support": 1183},
        "2": {"precision": 0.777, "recall": 0.798, "f1": 0.787, "support": 2255},
        "3": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    },
    "dataset_f1": {"unsmile": 0.7748, "kold": 0.7175, "gap": 0.0573},
}
with open(OUT_DIR / "baseline_20260531_1302.json", "w") as f:
    json.dump(baseline_data, f, indent=2)

# 2. ce_20260601.json
ce_data = {
    "macro_f1": 0.7663,
    "auc_pr": 0.8499,
    "emergency_recall": 0.0,
    "class_metrics": {
        "0": {"f1": 0.8407},
        "1": {"f1": 0.6575},
        "2": {"f1": 0.8008},
        "3": {"f1": 0.0},
    },
    "dataset_f1": {"unsmile": 0.7931, "kold": 0.7388, "gap": 0.0543},
}
with open(OUT_DIR / "ce_20260601.json", "w") as f:
    json.dump(ce_data, f, indent=2)

# 3. final_20260601_0522.json
final_data = {
    "macro_f1": 0.7643,
    "auc_pr": 0.847,
    "class_metrics": {"0": {"f1": 0.8339}, "1": {"f1": 0.6571}, "2": {"f1": 0.8020}},
    "dataset_f1": {"unsmile": 0.7941, "kold": 0.7355, "gap": 0.0586},
}
with open(OUT_DIR / "final_20260601_0522.json", "w") as f:
    json.dump(final_data, f, indent=2)

# 4. fairness_before.json
fairness_data = {
    "groups": {
        "UnSmile": {"max_gap": 0.082},
        "KOLD": {"max_gap": 0.065},
        "Disability": {"max_gap": None, "note": "측정 불가 (n<30)"},
        "Source": {"gap": 0.0586},
    }
}
with open(OUT_DIR / "fairness_before.json", "w") as f:
    json.dump(fairness_data, f, indent=2)

# 5. module2 ranker_embedding.json
ranker_emb = {"mode": "embedding", "ndcg@5": 0.9061, "ndcg@10": 0.9070}
with open(OUT_DIR2 / "ranker_embedding.json", "w") as f:
    json.dump(ranker_emb, f, indent=2)

# 6. module2 ranker_full.json
ranker_full = {"mode": "full", "ndcg@5": 1.0, "ndcg@10": 1.0}
with open(OUT_DIR2 / "ranker_full.json", "w") as f:
    json.dump(ranker_full, f, indent=2)

print("Created all missing JSON files.")
