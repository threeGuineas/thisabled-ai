from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 한글 폰트 설정 (Mac)
plt.rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_initial_imbalance():
    labels = ["정상(0)", "주의(1)", "경고(2)", "긴급(3)"]
    counts = [24795, 11826, 22550, 0]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=colors)
    plt.title("초기 시드 데이터 라벨 분포 (극단적 불균형)", fontsize=16)
    plt.ylabel("데이터 수", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}건\n({height/sum(counts)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "initial_imbalance.png", dpi=300)
    plt.close()


def plot_model_progression():
    stages = ["Baseline (Focal)", "CE+α", "최종 (CE+α+합성)"]
    macro_f1 = [0.7464, 0.7663, 0.7643]
    caution_f1 = [0.6246, 0.6575, 0.6571]
    emergency_recall = [0.0, 0.0, 1.0]

    x = np.arange(len(stages))
    width = 0.25

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width, macro_f1, width, label="Macro-F1", color="#3498db")
    bars2 = plt.bar(x, caution_f1, width, label="주의(1) F1", color="#f1c40f")
    bars3 = plt.bar(x + width, emergency_recall, width, label="긴급(3) Recall", color="#e74c3c")

    plt.ylabel("Score", fontsize=12)
    plt.title("불균형 처리 전략에 따른 성능 추이", fontsize=16)
    plt.xticks(x, stages, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc="upper left", fontsize=11)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_progression.png", dpi=300)
    plt.close()


def plot_confusion_matrix_final():
    # Baseline matrix from report, assumed similar for final but with recall 1.0 for emergency on synthetic holdout.
    # To be fully accurate to the final model (CE+alpha) on seed test:
    # We will just show the baseline matrix as a representation or approximate the CE one.
    # Since we don't have the exact CE confusion matrix, let's plot the baseline one as 'module1_confusion_baseline.png'
    # which is referenced in baseline.md
    cm = np.array([[2023, 193, 264, 0], [189, 742, 252, 0], [198, 258, 1799, 0], [0, 0, 0, 0]])
    labels = ["정상(0)", "주의(1)", "경고(2)", "긴급(3)"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Baseline / Seed Test)", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "module1_confusion_baseline.png", dpi=300)
    plt.close()


def plot_ndcg_comparison():
    modes = ["Embedding Mode (Leakage-free)", "Full Mode (Sanity Check)"]
    ndcg5 = [0.9061, 1.0000]
    ndcg10 = [0.9070, 1.0000]

    x = np.arange(len(modes))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, ndcg5, width, label="NDCG@5", color="#2ecc71")
    plt.bar(x + width / 2, ndcg10, width, label="NDCG@10", color="#27ae60")

    plt.ylabel("NDCG Score", fontsize=12)
    plt.title("모듈 ② LambdaMART 성능 (Cold-start)", fontsize=16)
    plt.xticks(x, modes, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=11)

    for i in range(len(modes)):
        plt.text(
            x[i] - width / 2,
            ndcg5[i] + 0.01,
            f"{ndcg5[i]:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
        plt.text(
            x[i] + width / 2,
            ndcg10[i] + 0.01,
            f"{ndcg10[i]:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ndcg_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_initial_imbalance()
    plot_model_progression()
    plot_confusion_matrix_final()
    plot_ndcg_comparison()
    print("Generated basic figures.")
