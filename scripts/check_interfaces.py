"""Kiểm tra giao diện giữa các phần của Người 1, 2, 3 (dùng khi review PR, ngày 3-4).

    python scripts/check_interfaces.py

Khác pytest: không kiểm tra công thức mà kiểm tra các phần GHÉP được với train.py/evaluate.py
(shape, dtype, thiết bị, kiểu trả về). Phần nào còn NotImplementedError thì báo "chưa làm" và bỏ qua.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CFG = OmegaConf.load(ROOT / "configs/default.yaml")
results = []


def check(name):
    def deco(fn):
        try:
            fn()
            results.append((name, "OK", ""))
        except NotImplementedError as e:
            results.append((name, "chưa làm", str(e)))
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-1]
            results.append((name, "LỖI", f"{type(e).__name__}: {e}  ({Path(tb.filename).name}:{tb.lineno})"))
        return fn
    return deco


@check("Người 1 - make_env: quan sát (4,84,84) uint8, thưởng float, info có x_pos/flag_get")
def _env():
    from mario_ppo.env import make_env
    env_cfg = OmegaConf.to_container(CFG.env)
    for eval_mode in (False, True):
        env = make_env("1-1", seed=0, eval_mode=eval_mode, **env_cfg)
        try:
            obs, _ = env.reset(seed=0)
            assert env.observation_space.shape == (4, 84, 84), env.observation_space
            assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8, (obs.shape, obs.dtype)
            assert env.action_space.n == 7
            for _ in range(5):
                obs, r, term, trunc, info = env.step(1)
            assert np.ndim(r) == 0, f"thưởng phải là số, nhận shape {np.shape(r)}"
            assert "vector_reward" in info and np.shape(info["vector_reward"]) == (5,)
            assert {"x_pos", "flag_get"} <= set(info)
            assert 0 < obs.mean() < 255, "ảnh toàn đen/trắng?"
        finally:
            env.close()


@check("Người 2 - ActorCritic: forward/act đúng shape, dtype, nhận uint8")
def _model():
    from mario_ppo.model import ActorCritic
    m = ActorCritic(7, in_channels=4)
    obs = torch.randint(0, 256, (16, 4, 84, 84), dtype=torch.uint8)
    logits, value = m(obs)
    assert logits.shape == (16, 7) and logits.dtype == torch.float32, (logits.shape, logits.dtype)
    assert value.shape == (16,) and value.dtype == torch.float32, (value.shape, value.dtype)
    a, logp, v = m.act(obs)
    assert a.shape == (16,) and a.dtype == torch.long and logp.shape == (16,) and v.shape == (16,)
    n = sum(p.numel() for p in m.parameters())
    print(f"    số tham số: {n:,}")


@check("Người 3 - compute_gae: (T,N) vào -> (T,N) ra, returns = adv + values")
def _gae():
    from mario_ppo.ppo import compute_gae
    T, N = 256, 16
    z = torch.zeros(T, N)
    adv, ret = compute_gae(torch.randn(T, N), torch.randn(T, N), z, z, z, torch.randn(N), 0.9, 0.95)
    assert adv.shape == (T, N) and ret.shape == (T, N)


@check("Người 2+3 - ppo_update chạy trên batch giống train.py, trả đủ khóa progress.csv")
def _update():
    from mario_ppo.model import ActorCritic
    from mario_ppo.ppo import ppo_update
    m = ActorCritic(7)
    opt = torch.optim.Adam(m.parameters(), lr=2.5e-4, eps=1e-5)
    B = 512
    batch = {"obs": torch.randint(0, 256, (B, 4, 84, 84), dtype=torch.uint8),
             "actions": torch.randint(0, 7, (B,)), "old_logp": torch.full((B,), -np.log(7)),
             "old_values": torch.zeros(B), "advantages": torch.randn(B), "returns": torch.randn(B)}
    a = OmegaConf.merge(CFG.algo, {"update_epochs": 1, "num_minibatches": 2})
    stats = ppo_update(m, opt, batch, a)
    need = {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"}
    assert need <= set(stats), f"thiếu {need - set(stats)}"
    assert all(np.isfinite(float(v)) for v in stats.values()), stats


if __name__ == "__main__":
    print()
    for name, status, msg in results:
        mark = {"OK": "[OK]  ", "chưa làm": "[--]  ", "LỖI": "[LỖI]"}[status]
        print(f"{mark} {name}" + (f"\n        {msg}" if status == "LỖI" else ""))
    sys.exit(1 if any(s == "LỖI" for _, s, _ in results) else 0)
