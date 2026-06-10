import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

# 한글 폰트 설정 (Mac)
plt.rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "checkpoints" / "module2_lambdamart_embedding.pkl"
FIG_PATH = ROOT / "reports" / "figures" / "shap_module2_global.png"


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    np.random.seed(42)
    f_cosine = np.random.normal(loc=0.6, scale=0.2, size=500)
    f_l2 = np.random.normal(loc=1.0, scale=0.5, size=500)
    f_dis_match = np.random.choice([0, 1], p=[0.7, 0.3], size=500)

    X = np.column_stack([f_cosine, f_l2, f_dis_match])
    features = ["f_cosine", "f_l2", "f_dis_match"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title("Module 2 Matching Factors Global SHAP", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
