"""Tóm tắt một thư mục run: SPS, dòng progress cuối, checkpoint tốt nhất theo evals.csv.

    python scripts/run_summary.py runs/1-1/run1             # in tóm tắt
    python scripts/run_summary.py runs/1-1/run1 --best-ckpt # chỉ in đường dẫn checkpoint tốt nhất
"""
import argparse
import csv
import sys
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def best_checkpoint(run):
    """Checkpoint có flag_rate cao nhất trong evals.csv (hòa thì lấy bước muộn hơn); không có evals thì lấy cái mới nhất."""
    ckpts = run / "checkpoints"
    rows = [r for r in read_csv(run / "evals.csv") if (ckpts / f"step_{int(r['global_step']):09d}.pt").exists()]
    if rows:
        r = max(rows, key=lambda r: (float(r["flag_rate"]), int(r["global_step"])))
        return ckpts / f"step_{int(r['global_step']):09d}.pt", float(r["flag_rate"])
    files = sorted(ckpts.glob("step_*.pt"))
    return (files[-1], None) if files else (None, None)


def _total_steps(run):
    cfg = run / "config.yaml"
    if not cfg.exists():
        return None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("total_steps:"):
            return float(line.split(":", 1)[1].strip().replace("_", ""))
    return None


def _warnings(progress):
    """Các dấu hiệu cần can thiệp (ngày 7): kẹt ở ống nước, KL quá lớn, entropy sụp."""
    last = progress[-1]
    step = float(last["global_step"])
    x = float(last["x_pos"]) if last.get("x_pos") not in (None, "", "nan") else None
    warn = []
    if x is not None and step >= 1_000_000 and x < 400:
        warn.append(f"x_pos TB = {x:.0f} < 400 sau {step / 1e6:.1f}M bước: Mario kẹt ở ống nước đầu tiên "
                    "-> cân nhắc bật NoProgressTruncation (Người 1) và chạy Run 2")
    if len(progress) >= 10:
        recent = [float(r["x_pos"]) for r in progress[-10:] if r.get("x_pos") not in (None, "", "nan")]
        older = [float(r["x_pos"]) for r in progress[-60:-50] if r.get("x_pos") not in (None, "", "nan")]
        if recent and older and step >= 2_000_000 and max(recent) <= sum(older) / len(older):
            warn.append("x_pos không tăng trong ~50 lần cập nhật gần nhất (chững lại)")
    kl = float(last.get("approx_kl") or 0)
    if kl > 0.05:
        warn.append(f"approx_kl = {kl:.3f} > 0.05: bước cập nhật quá lớn -> giảm lr hoặc đặt algo.target_kl=0.03")
    ent = last.get("entropy")
    if ent not in (None, "") and float(ent) < 0.3:
        warn.append(f"entropy = {float(ent):.2f}: chính sách gần tất định, có thể đã sụp sớm -> tăng ent_coef")
    for w in warn:
        print(f"  CẢNH BÁO: {w}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--best-ckpt", action="store_true")
    args = p.parse_args()
    if not args.run.is_dir():
        sys.exit(f"không tìm thấy thư mục run: {args.run}")

    ckpt, flag = best_checkpoint(args.run)
    if args.best_ckpt:
        if ckpt is None:
            sys.exit("chưa có checkpoint nào")
        print(ckpt.as_posix())
        return

    progress = read_csv(args.run / "progress.csv")
    print(f"run: {args.run.as_posix()}")
    if progress:
        sps = [float(r["sps"]) for r in progress]
        last = progress[-1]
        print(f"  số lần cập nhật : {len(progress)}")
        print(f"  global_step     : {last['global_step']}")
        print(f"  SPS trung bình  : {sum(sps) / len(sps):.1f}   (cuối: {sps[-1]:.1f})")
        print(f"  ep_return       : {last['ep_return']}   x_pos: {last['x_pos']}   flag_rate: {last['flag_rate']}")
        if "approx_kl" in last:
            print(f"  approx_kl       : {last['approx_kl']}   entropy: {last.get('entropy', '?')}")
        remaining = float(progress[-1].get("global_step", 0))
        total = _total_steps(args.run)
        if total and sps[-1] > 0 and remaining < total:
            print(f"  còn lại ước tính: {(total - remaining) / sps[-1] / 3600:.1f} giờ (tới {total:,.0f} bước)")
        _warnings(progress)
    else:
        print("  chưa có progress.csv")
    evals = read_csv(args.run / "evals.csv")
    if evals:
        print("  đánh giá theo checkpoint (global_step: flag_rate):")
        for r in evals:
            print(f"    {int(r['global_step']):>10d}: {float(r['flag_rate']):.2f}")
    if ckpt is not None:
        extra = f" (flag_rate {flag:.2f})" if flag is not None else ""
        print(f"  checkpoint tốt nhất: {ckpt.as_posix()}{extra}")


if __name__ == "__main__":
    main()
