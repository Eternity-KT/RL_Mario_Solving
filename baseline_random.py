"""Benchmark the environment with a random agent, without loading a network."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import psutil
from PIL import Image

from src.env import make_mario_env

ROOT = Path(__file__).resolve().parent


def run_baseline(
    config: dict,
    *,
    steps: int = 10_000,
    seed: int = 42,
    warmup_steps: int = 100,
    capture: Path | None = None,
) -> dict:
    """Measure decision throughput, emulator throughput and process RSS.

    Initialization and warmup are excluded from elapsed_seconds. Episode resets
    during the measured loop are included. RSS includes Python and dependencies,
    and is sampled every 256 decisions; it is not an exact peak or a leak test.
    """
    if steps < 1 or warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps non-negative")
    config = dict(config)
    if capture is not None:
        if config.get("render_mode") == "human":
            raise ValueError("Use rgb_array, not human rendering, with --capture")
        config["render_mode"] = "rgb_array"

    process = psutil.Process()
    with make_mario_env(**config) as env:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        for _ in range(warmup_steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if config.get("render_mode") == "human":
                env.render()
            if terminated or truncated:
                env.reset()

        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        rss_start = process.memory_info().rss
        rss_samples = [rss_start]
        frames = episodes = wins = terminations = truncations = 0
        total_reward = 0.0
        max_x = int(info.get("x_pos", 0))
        started = perf_counter()
        for step in range(1, steps + 1):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if config.get("render_mode") == "human":
                env.render()
            frames += int(info["frames_this_step"])
            total_reward += float(reward)
            max_x = max(max_x, int(info.get("x_pos", 0)))
            if step % 256 == 0:
                rss_samples.append(process.memory_info().rss)
            if terminated or truncated:
                episodes += 1
                wins += int(bool(info.get("flag_get", False)))
                terminations += int(terminated)
                truncations += int(truncated)
                if step < steps:
                    observation, info = env.reset()
        elapsed = perf_counter() - started
        rss_end = process.memory_info().rss
        rss_samples.append(rss_end)
        if capture is not None:
            capture.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render()).save(capture)

        return {
            "config": config,
            "seed": seed,
            "warmup_decisions": warmup_steps,
            "agent_decisions": steps,
            "emulator_frames": frames,
            "elapsed_seconds": elapsed,
            "agent_steps_per_second": steps / elapsed,
            "emulator_frames_per_second": frames / elapsed,
            "episodes_completed": episodes,
            "wins": wins,
            "terminated_episodes": terminations,
            "truncated_episodes": truncations,
            "total_default_reward": total_reward,
            "max_x": max_x,
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "action_count": int(env.action_space.n),
            "rss_start_mib": rss_start / 1024**2,
            "rss_end_mib": rss_end / 1024**2,
            "rss_peak_sampled_mib": max(rss_samples) / 1024**2,
            "rss_sample_count": len(rss_samples),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "gym-super-mario-bros", "nes-py", "gymnasium", "numpy", "Pillow"
                )
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.json")
    parser.add_argument("--steps", type=int, default=10_000, help="Agent decisions")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-type", choices=("SIMPLE_MOVEMENT", "RIGHT_ONLY", "RIGHT_JUMP")
    )
    parser.add_argument("--render", action="store_true", help="Open a native game window")
    parser.add_argument("--capture", type=Path, help="Save a native RGB frame as PNG")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline_random.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0:
        parser.error("--steps must be positive and --warmup-steps non-negative")
    if args.render and args.capture:
        parser.error("--render and --capture cannot be used together")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.action_type:
        config["action_type"] = args.action_type
    if args.render:
        config["render_mode"] = "human"
    report = run_baseline(
        config, steps=args.steps, seed=args.seed,
        warmup_steps=args.warmup_steps, capture=args.capture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['agent_decisions']:,} decisions / {report['emulator_frames']:,} frames; "
        f"{report['agent_steps_per_second']:.1f} decisions/s; "
        f"{report['emulator_frames_per_second']:.1f} emulator frames/s"
    )
    print(
        f"Episodes: {report['episodes_completed']}; max X: {report['max_x']}; "
        f"sampled peak RSS: {report['rss_peak_sampled_mib']:.1f} MiB"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
