"""Bài kiểm tra BỔ SUNG của nhóm cho phần tích hợp (không thuộc bộ công khai của đề; Người 4 quản lý).

Bịt các chỗ mà tests/test_pipeline.py không bắt được:
  - môi trường đánh giá vẫn bật NOOP ngẫu nhiên / chuẩn hóa phần thưởng (test công khai chỉ xem wrapper ngoài cùng);
  - phần thưởng khi đánh giá phải là w · r thô; không gian quan sát khớp quan sát thật;
  - nội dung progress.csv, evals.csv, config.yaml, checkpoint (có optimizer);
  - chạy tiếp (resume) từ checkpoint: khôi phục trọng số, số bước, lịch giảm tốc độ học;
  - evaluate tái lập được với cùng seed; dòng lệnh của train (load_config) và evaluate (main).
Chạy trên trình giả lập thật, mất khoảng 15-30 giây.
"""
import csv
import json
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mario_ppo.env import DEFAULT_WEIGHTS, make_env
from mario_ppo.ppo import linear_schedule
from mario_ppo.wrappers import RandomNoopStart


def _wrapper_types(env):
    types = []
    while isinstance(env, gym.Wrapper):
        types.append(type(env))
        env = env.env
    return types


@pytest.mark.parametrize("eval_mode", [False, True])
def test_noop_start_and_reward_normalisation_only_when_training(eval_mode):
    env = make_env("1-1", seed=0, noop_max=30, normalize_reward=True, eval_mode=eval_mode)
    try:
        types = _wrapper_types(env)
        assert (RandomNoopStart in types) == (not eval_mode)
        assert (gym.wrappers.NormalizeReward in types) == (not eval_mode)
    finally:
        env.close()


def test_eval_reward_is_raw_weighted_sum():
    env = make_env("1-1", seed=0, eval_mode=True)
    try:
        env.reset(seed=0)
        for _ in range(10):
            _, reward, terminated, truncated, info = env.step(1)
            assert np.ndim(reward) == 0
            assert float(reward) == pytest.approx(float(np.dot(DEFAULT_WEIGHTS, info["vector_reward"])), abs=1e-5)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_make_env_observation_space_matches_observations():
    env = make_env("1-1", seed=0)
    try:
        space = env.observation_space
        assert space.shape == (4, 84, 84) and space.dtype == np.uint8
        obs, _ = env.reset(seed=0)
        assert space.contains(obs)
        obs, reward, *_, info = env.step(1)
        assert space.contains(obs)
        assert np.ndim(reward) == 0
        assert np.shape(info["vector_reward"]) == (5,)
        assert 0 < obs.mean() < 255, "ảnh toàn đen / toàn trắng?"
    finally:
        env.close()


# ------------------------------------------------------------------------------------------------
# Huấn luyện tí hon (1 môi trường, 4 lần cập nhật x 16 bước), dùng chung cho các test bên dưới
# ------------------------------------------------------------------------------------------------

def _cfg(run_dir, **over):
    cfg = OmegaConf.load("configs/default.yaml")
    settings = {"run_dir": str(run_dir), "algo.num_envs": 1, "algo.num_steps": 16, "algo.num_minibatches": 2,
                "algo.update_epochs": 1, "algo.total_steps": 64, "checkpoint_every": 32, "eval_episodes": 1,
                "eval_max_steps": 20, "video": False, "torch_threads": 1}
    settings.update(over)
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from mario_ppo.train import train
    return train(_cfg(tmp_path_factory.mktemp("pipeline") / "run"))


def test_progress_csv_has_one_finite_row_per_update(trained):
    cfg = OmegaConf.load(trained / "config.yaml")
    assert cfg.algo.total_steps == 64 and cfg.algo.num_envs == 1  # config.yaml lưu đúng cấu hình đã chạy
    rows = _rows(trained / "progress.csv")
    assert [int(r["global_step"]) for r in rows] == [16, 32, 48, 64]
    need = {"sps", "lr", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "minibatch_updates",
            "episodes", "ep_return", "ep_length", "x_pos", "flag_rate"}
    assert need <= set(rows[0]), f"thiếu cột {need - set(rows[0])}"
    for r in rows:
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
            assert math.isfinite(float(r[k])), (r["global_step"], k, r[k])
        assert int(float(r["minibatch_updates"])) == cfg.algo.num_minibatches * cfg.algo.update_epochs
    expected_lr = [linear_schedule(cfg.algo.lr, s, 64) if cfg.algo.anneal_lr else cfg.algo.lr for s in (0, 16, 32, 48)]
    assert [float(r["lr"]) for r in rows] == pytest.approx(expected_lr)


def test_checkpoints_and_evals_written(trained):
    names = sorted(p.name for p in (trained / "checkpoints").glob("*.pt"))
    assert names == ["step_000000032.pt", "step_000000064.pt"]
    rows = _rows(trained / "evals.csv")
    assert [int(r["global_step"]) for r in rows] == [32, 64]
    assert all(0.0 <= float(r["flag_rate"]) <= 1.0 for r in rows)
    ck = torch.load(trained / "checkpoints" / names[-1], weights_only=False)
    assert {"model", "optimizer", "config", "global_step"} <= set(ck)
    assert ck["global_step"] == 64 and ck["config"]["algo"]["total_steps"] == 64


def test_resume_restores_weights_step_and_lr_schedule(trained, tmp_path):
    from mario_ppo.train import train
    ckpt = trained / "checkpoints" / "step_000000064.pt"
    ck0 = torch.load(ckpt, weights_only=False)

    # Đã đủ số bước: không huấn luyện thêm, checkpoint ghi ra phải giữ nguyên trọng số đã nạp.
    same = train(_cfg(tmp_path / "same", resume=str(ckpt), eval_episodes=0))
    ck1 = torch.load(same / "checkpoints" / "step_000000064.pt", weights_only=False)
    assert ck1["global_step"] == 64
    for k, v in ck0["model"].items():
        assert torch.equal(ck1["model"][k], v), k

    # Chạy tiếp tới 128 bước: số bước nối tiếp, tốc độ học tiếp tục giảm từ bước 64.
    more = train(_cfg(tmp_path / "more", resume=str(ckpt), eval_episodes=0, **{"algo.total_steps": 128}))
    rows = _rows(more / "progress.csv")
    assert [int(r["global_step"]) for r in rows] == [80, 96, 112, 128]
    cfg = OmegaConf.load(more / "config.yaml")
    if cfg.algo.anneal_lr:
        assert float(rows[0]["lr"]) == pytest.approx(linear_schedule(cfg.algo.lr, 64, 128))
    names = sorted(p.name for p in (more / "checkpoints").glob("*.pt"))
    assert names == ["step_000000096.pt", "step_000000128.pt"]


def test_evaluate_is_reproducible_and_cli_writes_json(trained, tmp_path):
    from mario_ppo.evaluate import evaluate_checkpoint, main
    ckpt = trained / "checkpoints" / "step_000000064.pt"
    a = evaluate_checkpoint(ckpt, episodes=2, seed=7, max_steps=25)
    b = evaluate_checkpoint(ckpt, episodes=2, seed=7, max_steps=25)
    assert a["level"] == "1-1"  # không truyền level -> lấy màn trong cấu hình của checkpoint
    for k in ("noop_steps", "x_pos", "steps", "flag_rate"):
        assert a[k] == b[k], k
    out = tmp_path / "res.json"
    main(["--ckpt", str(ckpt), "--episodes", "1", "--max-steps", "10", "--out", str(out)])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["episodes"] == 1 and len(res["x_pos"]) == 1 and isinstance(res["solved"], bool)


def test_load_config_merges_command_line_overrides():
    from mario_ppo.train import load_config
    default = OmegaConf.load("configs/default.yaml")
    cfg = load_config(["algo.lr=1e-4", "level=1-2", "env.noop_max=0"])
    assert isinstance(cfg.algo.lr, float) and cfg.algo.lr == pytest.approx(1e-4)
    assert cfg.level == "1-2" and cfg.env.noop_max == 0
    assert cfg.algo.num_envs == default.algo.num_envs  # các khóa khác giữ nguyên
