"""Bài kiểm tra BỔ SUNG của nhóm cho ppo.py (không thuộc bộ công khai của đề; Người 4 quản lý).

Bịt các chỗ mà tests/test_ppo.py không bắt được (đã thử cài sai mà test công khai vẫn PASS):
  - ppo_loss quên nhân vf_coef (test công khai chỉ dùng vf_coef = 1, còn cấu hình thật là 0.25);
  - ppo_loss bỏ qua clip_vloss;
  - ppo_update quên zero_grad (gradient cộng dồn qua các mini-batch), quên clip_grad_norm_;
  - ppo_update không xáo trộn mỗi epoch / không phủ đủ mọi mẫu;
  - stats lấy của mini-batch cuối thay vì trung bình; dừng sớm theo approx_kl từng mini-batch
    thay vì trung bình của epoch.
Quy ước của nhóm (xem NGUOI_3): biến thể cắt của value loss dùng cfg.clip_coef.
"""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import mario_ppo.ppo as ppo


def _cfg(**kw):
    base = dict(clip_coef=0.2, vf_coef=1.0, ent_coef=0.01, norm_adv=False, clip_vloss=False,
                update_epochs=1, num_minibatches=1, max_grad_norm=0.5, target_kl=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _hand_batch(v=2.2):
    """Mini-batch tính tay như test_ppo.py: pi_old = (2/3; 1/3), v_old = 2, pi_theta = (0,75; 0,25), v(s,w) = v."""
    old_logp = torch.log(torch.tensor([2 / 3, 2 / 3, 1 / 3]))
    actions = torch.tensor([0, 0, 1])
    theta = torch.tensor([math.log(3.0), 0.0], requires_grad=True)
    dist = torch.distributions.Categorical(logits=theta.expand(3, 2))
    values = torch.tensor(v, requires_grad=True).expand(3)
    batch = dict(old_logp=old_logp, advantages=torch.tensor([1.0, 1.0, -2.0]),
                 returns=torch.tensor([3.0, 3.0, 0.0]), old_values=torch.full((3,), 2.0))
    return dist.log_prob(actions), dist.entropy(), values, batch


def test_total_loss_uses_vf_coef():
    loss, stats = ppo.ppo_loss(*_hand_batch(), _cfg(vf_coef=0.25))
    entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert loss.item() == pytest.approx(-0.216667 + 0.25 * 2.04 - 0.01 * entropy, abs=1e-4)
    assert stats["value_loss"] == pytest.approx(2.04, abs=1e-4)  # stats là mất mát CHƯA nhân hệ số


@pytest.mark.parametrize("clip_vloss, expected", [(False, 2.25), (True, 2.51)])
def test_total_loss_respects_clip_vloss(clip_vloss, expected):
    # v = 2,5, v_old = 2, cắt 0,2 -> v_clip = 2,2; V_targ = (3; 3; 0)
    # không cắt: (0,25 + 0,25 + 6,25) / 3 = 2,25;  cắt: (max(0,25; 0,64) * 2 + max(6,25; 4,84)) / 3 = 2,51
    _, stats = ppo.ppo_loss(*_hand_batch(v=2.5), _cfg(clip_vloss=clip_vloss))
    assert stats["value_loss"] == pytest.approx(expected, abs=1e-4)


# ------------------------------------------------------------------------------------------------
# ppo_update
# ------------------------------------------------------------------------------------------------

class _TinyAC(nn.Module):
    """Actor-critic tí hon trên đặc trưng véc-tơ; ghi lại cột 0 của mọi đầu vào (dùng làm mã mẫu)."""

    def __init__(self, obs_dim=4, n_actions=3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh())
        self.pi, self.v = nn.Linear(32, n_actions), nn.Linear(32, 1)
        self.seen = []

    def forward(self, x):
        self.seen.append(x[:, 0].round().long().tolist())
        h = self.body(x.float())
        return self.pi(h), self.v(h).squeeze(-1)


class _RecordingSGD(torch.optim.SGD):
    """SGD với lr = 0 (tham số đứng yên); ghi lại gradient (đã cắt) ở mỗi lần step()."""

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.grads = []

    def step(self, closure=None):
        self.grads.append(torch.cat([p.grad.detach().flatten().clone()
                                     for g in self.param_groups for p in g["params"] if p.grad is not None]))
        return super().step(closure)


def _batch(model, obs):
    with torch.no_grad():
        logits, values = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.zeros(len(obs), dtype=torch.long)
        logp = dist.log_prob(actions)
    model.seen.clear()
    adv = torch.ones(len(obs))
    return dict(obs=obs, actions=actions, old_logp=logp, old_values=values, advantages=adv, returns=values + adv)


def _identical_batch(model, n=16):
    """Mọi mẫu giống nhau -> mọi mini-batch cho cùng một gradient (khi tham số đứng yên)."""
    return _batch(model, torch.ones(n, 4))


def test_ppo_update_zeroes_gradients_between_minibatches():
    torch.manual_seed(0)
    model = _TinyAC()
    opt = _RecordingSGD(model.parameters())
    ppo.ppo_update(model, opt, _identical_batch(model), _cfg(update_epochs=2, num_minibatches=2, max_grad_norm=1e9))
    assert len(opt.grads) == 4
    g0 = opt.grads[0]
    assert g0.abs().sum() > 0
    for g in opt.grads[1:]:
        # quên zero_grad -> gradient lần thứ k gấp k lần gradient đầu
        assert torch.allclose(g, g0, atol=1e-6)


def test_ppo_update_clips_gradient_norm():
    torch.manual_seed(0)
    model = _TinyAC()
    batch = _identical_batch(model)
    raw = _RecordingSGD(model.parameters())
    ppo.ppo_update(model, raw, batch, _cfg(max_grad_norm=1e9))
    max_norm = 0.1 * float(raw.grads[0].norm())  # ngưỡng chắc chắn nhỏ hơn chuẩn gradient thật
    clipped = _RecordingSGD(model.parameters())
    ppo.ppo_update(model, clipped, batch, _cfg(update_epochs=2, num_minibatches=2, max_grad_norm=max_norm))
    for g in clipped.grads:
        assert float(g.norm()) <= max_norm * (1 + 1e-4)


def test_ppo_update_shuffles_every_epoch_and_covers_every_sample():
    torch.manual_seed(0)
    model = _TinyAC()
    n, mbs, epochs = 64, 4, 3
    obs = torch.cat([torch.arange(n, dtype=torch.float32)[:, None], torch.randn(n, 3)], dim=1)
    batch = _batch(model, obs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ppo.ppo_update(model, opt, batch, _cfg(update_epochs=epochs, num_minibatches=mbs))
    calls = [ids for ids in model.seen if len(ids) == n // mbs]
    assert len(calls) == epochs * mbs, "mỗi mini-batch phải gọi model(obs) đúng một lần, kích thước B / num_minibatches"
    orders = [sum(calls[e * mbs:(e + 1) * mbs], []) for e in range(epochs)]
    for order in orders:
        assert sorted(order) == list(range(n)), "mỗi epoch phải dùng mỗi mẫu đúng một lần"
    assert tuple(range(n)) not in {tuple(o) for o in orders}, "không xáo trộn thứ tự mẫu"
    assert len({tuple(o) for o in orders}) == epochs, "phải xáo lại ở MỖI epoch"


def _patch_ppo_loss(monkeypatch, kl_values):
    """Thay stats của ppo_loss bằng số đã biết: mọi stats = số thứ tự lời gọi, approx_kl theo kl_values."""
    real = ppo.ppo_loss
    calls = []

    def fake(new_logp, entropy, values, batch, cfg):
        loss, stats = real(new_logp, entropy, values, batch, cfg)
        i = len(calls)
        calls.append(i)
        stats = {k: float(i) for k in stats}
        stats["approx_kl"] = kl_values[i % len(kl_values)]
        return loss, stats

    monkeypatch.setattr(ppo, "ppo_loss", fake)
    return calls


def _run_update(**cfg):
    torch.manual_seed(0)
    model = _TinyAC()
    batch = _batch(model, torch.randn(32, 4))
    return ppo.ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), batch, _cfg(**cfg))


def test_ppo_update_reports_mean_of_minibatch_stats(monkeypatch):
    _patch_ppo_loss(monkeypatch, [0.0])
    stats = _run_update(update_epochs=2, num_minibatches=4)
    assert stats["minibatch_updates"] == 8
    for k in ("policy_loss", "value_loss", "entropy", "clipfrac"):
        assert stats[k] == pytest.approx(3.5), k  # trung bình của 0, 1, ..., 7


@pytest.mark.parametrize("kl_values, target_kl, expected_updates", [
    ([0.6, 0.6, 0.6, 0.6], 0.5, 4),      # trung bình epoch 0,6 > 0,5 -> dừng sau epoch đầu
    ([1.0, 0.0, 0.0, 0.0], 0.5, 12),     # một mini-batch vượt nhưng trung bình 0,25 -> KHÔNG dừng
    ([0.0, 0.0, 0.0, 1.0], 0.5, 12),     # mini-batch CUỐI vượt nhưng trung bình 0,25 -> KHÔNG dừng
    ([100.0], None, 12),                 # target_kl = None -> không bao giờ dừng
])
def test_ppo_update_kl_early_stop_uses_epoch_mean(monkeypatch, kl_values, target_kl, expected_updates):
    _patch_ppo_loss(monkeypatch, kl_values)
    stats = _run_update(update_epochs=3, num_minibatches=4, target_kl=target_kl)
    assert stats["minibatch_updates"] == expected_updates
