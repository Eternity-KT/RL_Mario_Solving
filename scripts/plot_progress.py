"""Vẽ biểu đồ tiến độ huấn luyện từ progress.csv / evals.csv (một hoặc nhiều run để so sánh).

    python scripts/plot_progress.py runs/1-1/run1
    python scripts/plot_progress.py runs/1-1/run1 runs/1-1/run2 --labels "gốc" "NoProgress" --out results/so-sanh.png
    python scripts/plot_progress.py runs/1-1/gamma09 runs/1-1/gamma099 --dpi 300   # hình cho báo cáo

Mặc định lưu <run>/progress.png (một run) hoặc results/compare.png (nhiều run).
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [  # (cột, tiêu đề)
    ("ep_return", "Lợi tức lượt (thô, TB 100 lượt)"),
    ("x_pos", "x_pos cuối lượt (TB 100 lượt)"),
    ("flag_rate", "Tỉ lệ chạm cờ (100 lượt, khi train)"),
    ("entropy", "Entropy chính sách"),
    ("approx_kl", "approx_kl"),
    ("value_loss", "Value loss"),
    ("clipfrac", "Clip fraction"),
    ("sps", "SPS"),
]


def read(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for r in rows:
        for k, v in r.items():
            try:
                cols.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                pass
    return cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--labels", nargs="*")
    p.add_argument("--out", type=Path)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()
    labels = args.labels or [r.name for r in args.runs]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.ravel()
    for run, label in zip(args.runs, labels):
        prog = read(run / "progress.csv")
        if not prog:
            print(f"bỏ qua {run}: chưa có progress.csv")
            continue
        x = [s / 1e6 for s in prog["global_step"]]
        for ax, (col, title) in zip(axes, PANELS):
            if col in prog:
                ax.plot(x, prog[col], label=label, linewidth=1.2)
            ax.set_title(title)
        ev = read(run / "evals.csv")
        if ev:
            axes[8].plot([s / 1e6 for s in ev["global_step"]], ev["flag_rate"], marker="o", label=label)
    axes[8].set_title("flag_rate đánh giá ở checkpoint")
    axes[8].axhline(0.8, color="red", linestyle="--", linewidth=1, label="ngưỡng 0.8")
    axes[1].axhline(400, color="gray", linestyle=":", linewidth=1)  # ống nước đầu tiên ~x=400 (màn 1-1)
    for ax in axes:
        ax.set_xlabel("triệu bước")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[8].legend(fontsize=8)
    fig.tight_layout()

    out = args.out or (args.runs[0] / "progress.png" if len(args.runs) == 1 else Path("results/compare.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"đã lưu {out}")


if __name__ == "__main__":
    main()
