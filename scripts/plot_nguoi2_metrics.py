"""Script phân tích động học huấn luyện PPO và xuất 6 biểu đồ Màn 1-1 (Người 2 - Ngày 6–8).

Chức năng chính:
  1. Đọc trực tiếp `progress.csv` và `evals.csv` theo đúng lược đồ cột của `mario_ppo/train.py`.
  2. Chạy Bộ Chẩn Đoán Cảnh Báo Đỏ Tự Động (Mục 5.2 Kế hoạch tổng thể 14 ngày):
     - Phát hiện kẹt ống nước (`x_pos < 400` kéo dài sau 1,000,000 bước).
     - Phát hiện sụp đổ khám phá (`entropy < 0.1` sớm trước/sau 300,000 bước).
     - Phát hiện bước cập nhật chính sách quá lớn (`approx_kl > 0.05` liên tục).
     - Thống kê `SPS` trung bình, cực đại và ước tính thời gian huấn luyện.
  3. Xuất 6 biểu đồ đường cong học riêng lẻ + 1 bảng ghép lưới 2x3 chuẩn học thuật (300 DPI):
     (1) Episode Return, (2) x_pos, (3) flag_rate, (4) entropy, (5) approx_kl, (6) clipfrac.

Ví dụ chạy:
  python scripts/plot_nguoi2_metrics.py --run-dir runs/1-1/run1 --out-dir results/man_1_1
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ACTION_COUNTS = {
    "right_only": 5,
    "simple": 7,
    "simple_down": 8,
    "complex": 12,
}

REQUIRED_PROGRESS_COLUMNS = (
    "global_step",
    "sps",
    "lr",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "ep_return",
    "x_pos",
    "flag_rate",
)


def _to_float(raw: str | None) -> float:
    if raw is None:
        return math.nan
    s = raw.strip()
    if not s or s.lower() == "nan":
        return math.nan
    return float(s)


def load_progress_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Đọc `progress.csv` bằng thư viện chuẩn `csv` + `numpy` (không cần `pandas`)."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp nhật ký huấn luyện: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Tệp {csv_path} rỗng hoặc không có dòng tiêu đề cột.")

        missing = [c for c in REQUIRED_PROGRESS_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Tệp {csv_path} thiếu các cột bắt buộc theo `mario_ppo/train.py`: {missing}. "
                f"Các cột hiện có: {reader.fieldnames}"
            )

        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(_to_float(row.get(name)))

    if not columns["global_step"]:
        raise ValueError(f"Tệp {csv_path} chưa có dòng dữ liệu nào.")

    return {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}


def load_evals_csv(csv_path: Path) -> dict[str, np.ndarray] | None:
    """Đọc `evals.csv` (tùy chọn). Cột `x_pos` là chuỗi các số nguyên cách nhau bởi dấu cách."""
    if not csv_path.is_file():
        return None

    steps: list[float] = []
    flag_rates: list[float] = []
    mean_xpos: list[float] = []
    max_xpos: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        for row in reader:
            step = _to_float(row.get("global_step"))
            fr = _to_float(row.get("flag_rate"))
            raw_x = (row.get("x_pos") or "").strip()
            xs = [float(tok) for tok in raw_x.split()] if raw_x else []
            if math.isnan(step):
                continue
            steps.append(step)
            flag_rates.append(fr)
            mean_xpos.append(float(np.mean(xs)) if xs else math.nan)
            max_xpos.append(float(np.max(xs)) if xs else math.nan)

    if not steps:
        return None

    return {
        "global_step": np.asarray(steps, dtype=np.float64),
        "flag_rate": np.asarray(flag_rates, dtype=np.float64),
        "mean_x_pos": np.asarray(mean_xpos, dtype=np.float64),
        "max_x_pos": np.asarray(max_xpos, dtype=np.float64),
    }


def load_run_metadata(run_dir: Path, default_total_steps: int, default_n_actions: int) -> dict[str, Any]:
    """Đọc bổ sung `config.yaml` trong `run_dir` nếu có để tự động lấy `total_steps`, `level`, `n_actions`."""
    meta: dict[str, Any] = {
        "level": "1-1",
        "total_steps": default_total_steps,
        "n_actions": default_n_actions,
    }
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        return meta

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        if cfg.get("level") is not None:
            meta["level"] = str(cfg.level)
        if cfg.get("algo") and cfg.algo.get("total_steps") is not None:
            meta["total_steps"] = int(cfg.algo.total_steps)
        if cfg.get("env") and cfg.env.get("actions") in ACTION_COUNTS:
            meta["n_actions"] = ACTION_COUNTS[str(cfg.env.actions)]
    except Exception:
        pass
    return meta


def smooth_series(values: np.ndarray, window: int = 15, ema_alpha: float | None = None) -> np.ndarray:
    """Làm mượt chuỗi có hỗ trợ giá trị `NaN` bằng trung bình trượt (rolling mean) hoặc EMA."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    if ema_alpha is not None:
        alpha = float(np.clip(ema_alpha, 1e-4, 1.0))
        prev = math.nan
        for i, v in enumerate(arr):
            if math.isnan(v):
                out[i] = prev
            elif math.isnan(prev):
                prev = v
                out[i] = v
            else:
                prev = alpha * v + (1.0 - alpha) * prev
                out[i] = prev
        return out

    w = max(1, min(int(window), n))
    for i in range(n):
        start = max(0, i - w + 1)
        seg = arr[start : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) > 0:
            out[i] = float(np.mean(valid))
    return out


def _longest_true_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for flag in mask:
        if flag:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    total_sec = int(round(seconds))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}g {m:02d}p {s:02d}s"
    if m > 0:
        return f"{m}p {s:02d}s"
    return f"{s}s"


def run_diagnostics(
    progress: dict[str, np.ndarray],
    evals: dict[str, np.ndarray] | None,
    total_steps_target: int,
    n_actions: int,
) -> dict[str, Any]:
    """Bộ Chẩn Đoán Cảnh Báo Đỏ Tự Động (Nhiệm vụ Ngày 7 + Mục 5.2 Kế hoạch tổng thể)."""
    steps = progress["global_step"]
    sps = progress["sps"]
    x_pos = progress["x_pos"]
    entropy = progress["entropy"]
    approx_kl = progress["approx_kl"]
    clipfrac = progress["clipfrac"]
    flag_rate = progress["flag_rate"]
    ep_return = progress["ep_return"]

    last_step = float(steps[-1])
    first_step = float(steps[0])
    max_ent = math.log(n_actions)

    alerts: list[str] = []
    notes: list[str] = []

    # 1. Kiểm tra kẹt ống nước: x_pos < 400 kéo dài sau 1,000,000 bước
    after_1m = (steps >= 1_000_000) & (~np.isnan(x_pos))
    if np.any(after_1m):
        x_after_1m = x_pos[after_1m]
        stuck_mask = x_after_1m < 400.0
        stuck_ratio = float(np.mean(stuck_mask))
        stuck_streak = _longest_true_streak(stuck_mask)
        mean_x_after_1m = float(np.mean(x_after_1m))
        if stuck_streak >= 5 or stuck_ratio >= 0.30 or mean_x_after_1m < 400.0:
            alerts.append(
                f"[CẢNH BÁO ĐỎ - KẸT ỐNG NƯỚC] Sau 1,000,000 bước, `x_pos < 400` xuất hiện "
                f"{stuck_ratio * 100:.1f}% số cập nhật (chuỗi liên tiếp dài nhất: {stuck_streak}, "
                f"x_pos TB sau 1M: {mean_x_after_1m:.1f}).\n"
                f"   -> Đề xuất gửi Người 1 & Người 4: Bật `NoProgressTruncation(max_steps=200)` "
                f"bằng cờ `env.max_no_progress_steps=200`."
            )
        else:
            notes.append(
                f"[OK - TIẾN TRIỂN X_POS] Sau 1,000,000 bước, x_pos TB = {mean_x_after_1m:.1f} "
                f"(cực đại: {np.nanmax(x_pos):.1f}), không ghi nhận kẹt ống nước kéo dài."
            )
    else:
        valid_x = x_pos[~np.isnan(x_pos)]
        cur_mean_x = float(np.mean(valid_x[-10:])) if len(valid_x) else math.nan
        notes.append(
            f"[THEO DÕI - X_POS] Chưa cán mốc 1,000,000 bước (hiện tại: {int(last_step):,}). "
            f"x_pos TB 10 cập nhật gần nhất = {cur_mean_x:.1f}."
        )

    # 2. Kiểm tra sụp đổ khám phá: entropy < 0.1 sớm trước/sau 300,000 bước
    valid_ent_mask = ~np.isnan(entropy)
    ent_low_mask = valid_ent_mask & (entropy < 0.1)
    early_collapse = ent_low_mask & (steps <= 300_000)
    post_300k_collapse = ent_low_mask & (steps > 300_000)
    if np.any(early_collapse):
        first_bad_step = int(steps[early_collapse][0])
        min_ent = float(np.nanmin(entropy[early_collapse]))
        alerts.append(
            f"[CẢNH BÁO ĐỎ - SỤP ĐỔ KHÁM PHÁ SỚM] `entropy < 0.1` xuất hiện ngay từ bước "
            f"{first_bad_step:,} (trước mốc 300,000 bước; cực tiểu = {min_ent:.4f}, "
            f"tham chiếu khởi tạo ln({n_actions}) = {max_ent:.3f}).\n"
            f"   -> Đề xuất tinh chỉnh: Tăng `algo.ent_coef` lên `0.015` – `0.02` để tránh co cụm hành động sớm."
        )
    elif np.any(post_300k_collapse):
        first_bad_step = int(steps[post_300k_collapse][0])
        streak_low = _longest_true_streak(post_300k_collapse)
        ratio_low = float(np.mean(entropy[steps > 300_000] < 0.1))
        if streak_low >= 5 or ratio_low >= 0.25:
            alerts.append(
                f"[CẢNH BÁO ĐỎ - SỤP ĐỔ KHÁM PHÁ SAU 300K] `entropy < 0.1` kéo dài từ bước "
                f"{first_bad_step:,} ({ratio_low * 100:.1f}% số cập nhật sau 300k, chuỗi dài nhất: {streak_low}).\n"
                f"   -> Đề xuất tinh chỉnh: Tăng `algo.ent_coef` lên `0.015` – `0.02` (nếu `flag_rate` chưa đạt >= 0.8)."
            )
        else:
            notes.append(
                f"[LƯU Ý - ENTROPY] Có vài điểm `entropy < 0.1` thoáng qua sau 300k bước "
                f"(entropy cuối: {entropy[valid_ent_mask][-1]:.4f})."
            )
    else:
        ent_init = float(entropy[valid_ent_mask][0]) if np.any(valid_ent_mask) else math.nan
        ent_last = float(entropy[valid_ent_mask][-1]) if np.any(valid_ent_mask) else math.nan
        notes.append(
            f"[OK - ENTROPY] Chính sách duy trì khám phá ổn định (khởi đầu: {ent_init:.3f} / "
            f"ln({n_actions})={max_ent:.3f} -> hiện tại: {ent_last:.3f}, không rớt dưới 0.1)."
        )

    # 3. Kiểm tra bước cập nhật chính sách quá lớn: approx_kl > 0.05 liên tục
    valid_kl_mask = ~np.isnan(approx_kl)
    kl_high_mask = valid_kl_mask & (approx_kl > 0.05)
    kl_streak = _longest_true_streak(kl_high_mask)
    kl_ratio = float(np.mean(approx_kl[valid_kl_mask] > 0.05)) if np.any(valid_kl_mask) else 0.0
    kl_mean = float(np.nanmean(approx_kl))
    kl_max = float(np.nanmax(approx_kl))
    if kl_streak >= 3 or kl_ratio >= 0.10:
        first_kl_step = int(steps[kl_high_mask][0])
        alerts.append(
            f"[CẢNH BÁO ĐỎ - APPROX_KL QUÁ LỚN] `approx_kl > 0.05` xảy ra liên tục "
            f"(bắt đầu từ bước {first_kl_step:,}, chuỗi dài nhất: {kl_streak} lần cập nhật, "
            f"chiếm {kl_ratio * 100:.1f}%, cực đại: {kl_max:.4f}, TB: {kl_mean:.4f}).\n"
            f"   -> Đề xuất tinh chỉnh: Bật dừng sớm `algo.target_kl=0.02` hoặc giảm `algo.lr` (ví dụ `1.5e-4`)."
        )
    else:
        notes.append(
            f"[OK - APPROX_KL] Bước cập nhật chính sách ổn định (TB: {kl_mean:.4f}, cực đại: {kl_max:.4f}, "
            f"tỉ lệ > 0.05: {kl_ratio * 100:.1f}%, clipfrac TB: {float(np.nanmean(clipfrac)):.4f})."
        )

    # 4. Thống kê SPS & ước tính thời gian
    valid_sps = sps[~np.isnan(sps) & (sps > 0)]
    sps_mean = float(np.mean(valid_sps)) if len(valid_sps) else math.nan
    sps_median = float(np.median(valid_sps)) if len(valid_sps) else math.nan
    sps_max = float(np.max(valid_sps)) if len(valid_sps) else math.nan
    sps_min = float(np.min(valid_sps)) if len(valid_sps) else math.nan

    elapsed_est = (last_step / sps_mean) if (math.isfinite(sps_mean) and sps_mean > 0) else math.nan
    total_est = (total_steps_target / sps_mean) if (math.isfinite(sps_mean) and sps_mean > 0) else math.nan
    rem_steps = max(0.0, total_steps_target - last_step)
    eta_rem = (rem_steps / sps_mean) if (math.isfinite(sps_mean) and sps_mean > 0) else math.nan

    # 5. Tổng hợp evals.csv (nếu có)
    eval_summary = None
    if evals is not None and len(evals["global_step"]) > 0:
        best_idx = int(np.nanargmax(evals["flag_rate"]))
        eval_summary = {
            "num_evals": int(len(evals["global_step"])),
            "best_step": int(evals["global_step"][best_idx]),
            "best_flag_rate": float(evals["flag_rate"][best_idx]),
            "best_mean_x_pos": float(evals["mean_x_pos"][best_idx]),
            "solved_target_0_8": bool(evals["flag_rate"][best_idx] >= 0.8),
        }

    return {
        "num_rows": int(len(steps)),
        "first_step": int(first_step),
        "last_step": int(last_step),
        "total_steps_target": int(total_steps_target),
        "sps_mean": sps_mean,
        "sps_median": sps_median,
        "sps_max": sps_max,
        "sps_min": sps_min,
        "elapsed_est_sec": elapsed_est,
        "total_est_sec": total_est,
        "eta_rem_sec": eta_rem,
        "ep_return_last": float(ep_return[~np.isnan(ep_return)][-1]) if np.any(~np.isnan(ep_return)) else math.nan,
        "x_pos_max": float(np.nanmax(x_pos)) if np.any(~np.isnan(x_pos)) else math.nan,
        "flag_rate_max": float(np.nanmax(flag_rate)) if np.any(~np.isnan(flag_rate)) else math.nan,
        "alerts": alerts,
        "notes": notes,
        "eval_summary": eval_summary,
    }


def format_diagnostics_report(diag: dict[str, Any], run_dir: Path, level: str) -> str:
    lines = [
        "=" * 78,
        f"BÁO CÁO CHẨN ĐOÁN ĐỘNG HỌC HUẤN LUYỆN PPO — MÀN {level}",
        f"Thư mục nguồn: {run_dir}",
        "=" * 78,
        f"- Số lần cập nhật (dòng progress.csv): {diag['num_rows']:,}",
        f"- Tiến độ bước tác tử               : {diag['last_step']:,} / {diag['total_steps_target']:,} "
        f"({min(100.0, 100.0 * diag['last_step'] / max(1, diag['total_steps_target'])):.1f}%)",
        f"- Tốc độ SPS                        : TB = {diag['sps_mean']:.1f} | Trung vị = {diag['sps_median']:.1f} "
        f"| Cực đại = {diag['sps_max']:.1f} | Cực tiểu = {diag['sps_min']:.1f}",
        f"- Thời gian ước tính (theo SPS TB)  : Đã chạy ~{_format_duration(diag['elapsed_est_sec'])} | "
        f"Còn lại ~{_format_duration(diag['eta_rem_sec'])} | Tổng {diag['total_steps_target']:,} bước ~{_format_duration(diag['total_est_sec'])}",
        f"- Đỉnh huấn luyện (recent-100)      : x_pos max = {diag['x_pos_max']:.1f} | "
        f"flag_rate max = {diag['flag_rate_max']:.3f} | ep_return gần nhất = {diag['ep_return_last']:.2f}",
    ]

    if diag["eval_summary"] is not None:
        es = diag["eval_summary"]
        status = "ĐẠT NGƯỠNG VƯỢT MÀN (>= 0.8)" if es["solved_target_0_8"] else "CHƯA ĐẠT NGƯỠNG 0.8"
        lines.append(
            f"- Đánh giá định kỳ (evals.csv)      : {es['num_evals']} mốc | Tốt nhất tại bước {es['best_step']:,}: "
            f"flag_rate = {es['best_flag_rate']:.2f}, mean x_pos = {es['best_mean_x_pos']:.1f} [{status}]"
        )

    lines.append("-" * 78)
    lines.append("CHẨN ĐOÁN CẢNH BÁO ĐỎ (MỤC 5.2 KẾ HOẠCH TỔNG THỂ):")
    if diag["alerts"]:
        for alert in diag["alerts"]:
            lines.append(f"  {alert}")
    else:
        lines.append("  [KHÔNG CÓ CẢNH BÁO ĐỎ] Các chỉ số x_pos, entropy và approx_kl đều nằm trong vùng an toàn.")

    lines.append("-" * 78)
    lines.append("NHẬN XÉT CHI TIẾT:")
    for note in diag["notes"]:
        lines.append(f"  {note}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _step_formatter_and_label(max_step: float) -> tuple[float, str]:
    if max_step >= 1_000_000:
        return 1e6, "Bước tác tử (triệu bước)"
    if max_step >= 10_000:
        return 1e3, "Bước tác tử (nghìn bước)"
    return 1.0, "Bước tác tử"


def plot_learning_curves(
    progress: dict[str, np.ndarray],
    evals: dict[str, np.ndarray] | None,
    out_dir: Path,
    level: str,
    n_actions: int,
    smooth_window: int = 15,
    ema_alpha: float | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Xuất 6 file ảnh biểu đồ riêng lẻ + 1 bảng ghép lưới 2x3 chuẩn 300 DPI."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện `matplotlib` để xuất biểu đồ PNG. "
            "Cài đặt bằng `pip install matplotlib` hoặc chạy với cờ `--diagnostics-only`."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = progress["global_step"]
    scale, x_label = _step_formatter_and_label(float(np.nanmax(steps)))
    x_scaled = steps / scale
    max_ent = math.log(n_actions)
    smooth_label = f"EMA (alpha={ema_alpha})" if ema_alpha is not None else f"TB trượt (w={smooth_window})"

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    specs = [
        {
            "key": "ep_return",
            "filename": "01_ep_return.png",
            "title": "(a) Lợi tức lượt chơi (Episode Return)",
            "ylabel": "Episode Return (thô)",
            "color": "#1f77b4",
            "refs": [],
            "ylim": None,
            "eval_key": None,
        },
        {
            "key": "x_pos",
            "filename": "02_x_pos.png",
            "title": "(b) Tọa độ ngang trung bình (x_pos)",
            "ylabel": "Tọa độ x_pos (pixel)",
            "color": "#2ca02c",
            "refs": [(400.0, "Ngưỡng kẹt ống nước (400)", "#d62728", ":")],
            "ylim": None,
            "eval_key": "mean_x_pos",
        },
        {
            "key": "flag_rate",
            "filename": "03_flag_rate.png",
            "title": "(c) Tỷ lệ chạm cờ (flag_rate)",
            "ylabel": "Tỷ lệ chạm cờ",
            "color": "#ff7f0e",
            "refs": [(0.8, "Ngưỡng vượt màn (0.8)", "#d62728", "--")],
            "ylim": (-0.03, 1.05),
            "eval_key": "flag_rate",
        },
        {
            "key": "entropy",
            "filename": "04_entropy.png",
            "title": "(d) Policy Entropy",
            "ylabel": "Entropy H(π)",
            "color": "#9467bd",
            "refs": [
                (max_ent, f"Khởi tạo đều ln({n_actions}) ≈ {max_ent:.3f}", "#2ca02c", "--"),
                (0.1, "Cảnh báo sụp đổ (0.1)", "#d62728", ":"),
            ],
            "ylim": (0.0, max(max_ent * 1.08, float(np.nanmax(progress["entropy"])) * 1.05)),
            "eval_key": None,
        },
        {
            "key": "approx_kl",
            "filename": "05_approx_kl.png",
            "title": "(e) Độ lệch chính sách (approx_kl)",
            "ylabel": "Approx KL",
            "color": "#d62728",
            "refs": [(0.05, "Ngưỡng cảnh báo (0.05)", "#7f0000", "--")],
            "ylim": None,
            "eval_key": None,
        },
        {
            "key": "clipfrac",
            "filename": "06_clipfrac.png",
            "title": "(f) Tỷ lệ cắt PPO (clipfrac)",
            "ylabel": "Clip Fraction",
            "color": "#17becf",
            "refs": [],
            "ylim": (-0.02, None),
            "eval_key": None,
        },
    ]

    def _draw_panel(ax: Any, spec: dict[str, Any]) -> None:
        raw = progress[spec["key"]]
        smoothed = smooth_series(raw, window=smooth_window, ema_alpha=ema_alpha)
        valid_raw = ~np.isnan(raw)
        valid_sm = ~np.isnan(smoothed)

        if np.any(valid_raw):
            ax.plot(
                x_scaled[valid_raw],
                raw[valid_raw],
                color=spec["color"],
                alpha=0.22,
                linewidth=0.9,
                label="Dữ liệu gốc",
            )
        if np.any(valid_sm):
            ax.plot(
                x_scaled[valid_sm],
                smoothed[valid_sm],
                color=spec["color"],
                linewidth=2.0,
                label=smooth_label,
            )

        eval_key = spec["eval_key"]
        if evals is not None and eval_key is not None and eval_key in evals:
            ev_vals = evals[eval_key]
            ev_valid = ~np.isnan(ev_vals)
            if np.any(ev_valid):
                ax.scatter(
                    evals["global_step"][ev_valid] / scale,
                    ev_vals[ev_valid],
                    color="#111111",
                    marker="D",
                    s=28,
                    zorder=5,
                    label="Đánh giá định kỳ (evals.csv)",
                )

        for y_ref, ref_label, ref_color, ref_style in spec["refs"]:
            ax.axhline(y_ref, color=ref_color, linestyle=ref_style, linewidth=1.3, label=ref_label)

        ax.set_title(spec["title"])
        ax.set_xlabel(x_label)
        ax.set_ylabel(spec["ylabel"])
        if spec["ylim"] is not None:
            ax.set_ylim(spec["ylim"])
        ax.legend(loc="best", frameon=True, framealpha=0.85)

    saved_paths: list[Path] = []

    # 1. Xuất 6 biểu đồ riêng lẻ (300 DPI)
    for spec in specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=dpi)
        _draw_panel(ax, spec)
        fig.suptitle(f"Mario PPO — Màn {level}", fontsize=11, y=0.98, color="#333333")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        p = out_dir / spec["filename"]
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(p)

    # 2. Xuất bảng ghép lưới 2x3 tối ưu cho Mục 3 Báo cáo 8 trang (300 DPI)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), dpi=dpi)
    for ax, spec in zip(axes.flat, specs):
        _draw_panel(ax, spec)
    fig.suptitle(
        f"Đường cong học PPO trên Super Mario Bros — Màn {level}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    panel_path = out_dir / "learning_curves_6_panel.png"
    fig.savefig(panel_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(panel_path)

    return saved_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Đường dẫn thư mục run chứa `progress.csv` (và tùy chọn `evals.csv`, `config.yaml`).",
    )
    p.add_argument(
        "--out-dir",
        default=Path("results/man_1_1"),
        type=Path,
        help="Thư mục lưu 6 biểu đồ riêng lẻ, lưới 2x3 và báo cáo chẩn đoán (mặc định: results/man_1_1).",
    )
    p.add_argument("--smooth-window", type=int, default=15, help="Kích thước cửa sổ trung bình trượt (mặc định: 15).")
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="Nếu đặt (ví dụ 0.1), dùng làm mượt EMA thay cho trung bình trượt.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Độ phân giải ảnh xuất (mặc định: 300 DPI).")
    p.add_argument(
        "--total-steps",
        type=int,
        default=5_000_000,
        help="Ngân sách bước mục tiêu để ước tính ETA (tự ghi đè nếu có `config.yaml`).",
    )
    p.add_argument(
        "--n-actions",
        type=int,
        default=7,
        help="Số hành động của môi trường để tính ln(n_actions) (mặc định: 7 cho `simple`).",
    )
    p.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Chỉ chạy chẩn đoán số liệu ra màn hình, không xuất file ảnh PNG.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    progress = load_progress_csv(args.run_dir / "progress.csv")
    evals = load_evals_csv(args.run_dir / "evals.csv")
    meta = load_run_metadata(args.run_dir, args.total_steps, args.n_actions)

    diag = run_diagnostics(progress, evals, meta["total_steps"], meta["n_actions"])
    report_text = format_diagnostics_report(diag, args.run_dir, meta["level"])
    print(report_text)

    if not args.diagnostics_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.out_dir / "diagnostics_report.txt"
        report_path.write_text(report_text + "\n", encoding="utf-8")
        saved = plot_learning_curves(
            progress=progress,
            evals=evals,
            out_dir=args.out_dir,
            level=meta["level"],
            n_actions=meta["n_actions"],
            smooth_window=args.smooth_window,
            ema_alpha=args.ema_alpha,
            dpi=args.dpi,
        )
        print(f"\nĐã xuất báo cáo chẩn đoán: {report_path}")
        print(f"Đã xuất {len(saved)} biểu đồ ({args.dpi} DPI) tại: {args.out_dir}")
        for path in saved:
            print(f"  - {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
