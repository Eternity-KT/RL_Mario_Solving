# Bàn giao Người 2

Người 2 bàn giao kiến trúc mạng Actor-Critic cho Người 3 (`mario_ppo/ppo.py`) và Người 4 (`mario_ppo/train.py`, `mario_ppo/evaluate.py`) tích hợp vào pipeline PPO. Điểm vào là lớp [`ActorCritic`](../mario_ppo/model.py#L21) và hàm khởi tạo trực giao [`_init()`](../mario_ppo/model.py#L14).

## Đặc tả `ActorCritic`

Mạng sử dụng kiến trúc **Shared Nature CNN Backbone** kết hợp hai đầu ra độc lập: **Actor (`self.policy`)** sinh logits cho phân phối hành động và **Critic (`self.value`)** ước lượng giá trị trạng thái $V(s)$. Mạng chỉ nhận đầu vào thuần điểm ảnh `uint8` từ môi trường, tuyệt đối không sử dụng thông tin RAM hay `info` của trò chơi.

- **Khởi tạo:** `ActorCritic(n_actions, in_channels=4)` — hỗ trợ động số hành động (`n_actions`) và số khung hình xếp chồng (`in_channels`).
- **`forward(obs)`:** nhận `obs` kiểu `torch.uint8` shape `(B, in_channels, 84, 84)` trên CPU hoặc CUDA, chuẩn hóa một lần `x = obs.float() / 255.0` về $[0.0, 1.0]$, đi qua `self.trunk` và trả về `(logits, value)`:
  - `logits`: `torch.float32`, shape `(B, n_actions)` (không bọc Softmax).
  - `value`: `torch.float32`, shape `(B,)` (đã `.squeeze(-1)`, an toàn kể cả khi `B = 1`).
- **`act(obs)`:** giữ nguyên bản có sẵn với `@torch.no_grad()`, tự đẩy `obs.to(device)`, lấy mẫu từ `torch.distributions.Categorical(logits=logits)` và trả về `(action, logp, value)` đều nằm trên **CPU**, không giữ đồ thị đạo hàm.

## Kiến trúc và kích thước tensor từng tầng

Cấu hình mặc định: `in_channels = 4`, `n_actions = 7` (`actions: simple`).

| Tầng | Thành phần | Đầu vào | Đầu ra | Kernel / Stride | Kích hoạt | Khởi tạo (`gain`) | Số tham số |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Chuẩn hóa | `obs.float() / 255.0` | `(B, 4, 84, 84)` `uint8` | `(B, 4, 84, 84)` `float32` | — | — | — | 0 |
| `trunk.0` | `Conv2d(4, 32)` | `(B, 4, 84, 84)` | `(B, 32, 20, 20)` | `k=8, s=4, p=0` | `ReLU` | $\sqrt{2}$ | 8,224 |
| `trunk.2` | `Conv2d(32, 64)` | `(B, 32, 20, 20)` | `(B, 64, 9, 9)` | `k=4, s=2, p=0` | `ReLU` | $\sqrt{2}$ | 32,832 |
| `trunk.4` | `Conv2d(64, 64)` | `(B, 64, 9, 9)` | `(B, 64, 7, 7)` | `k=3, s=1, p=0` | `ReLU` | $\sqrt{2}$ | 36,928 |
| `trunk.6` | `Flatten()` | `(B, 64, 7, 7)` | `(B, 3136)` | — | — | — | 0 |
| `trunk.7` | `Linear(3136, 512)` | `(B, 3136)` | `(B, 512)` | — | `ReLU` | $\sqrt{2}$ | 1,606,144 |
| `policy` | `Linear(512, 7)` | `(B, 512)` | `(B, 7)` | — | — | `0.01` | 3,591 |
| `value` | `Linear(512, 1)` + `squeeze(-1)` | `(B, 512)` | `(B,)` | — | — | `1.0` | 513 |
| **Tổng** | **`ActorCritic(7)`** | `(B, 4, 84, 84)` | `(B, 7)` & `(B,)` | — | — | `bias = 0` | **1,688,232** |

## Vai trò hệ số `gain` trong khởi tạo trực giao `_init()`

Toàn bộ các lớp `Conv2d` và `Linear` được khởi tạo bằng `nn.init.orthogonal_(layer.weight, gain)` và `nn.init.zeros_(layer.bias)`:

- **`gain = np.sqrt(2)` (cho toàn bộ `self.trunk`):** bù phần phương sai bị cắt đi một nửa qua hàm kích hoạt `ReLU`, giữ tín hiệu lan truyền xuôi và gradient lan truyền ngược ổn định qua 3 tầng tích chập và tầng ẩn 512 chiều.
- **`gain = 0.01` (cho `self.policy`):** giữ biên độ logits ban đầu rất nhỏ quanh `0`, khiến `Softmax(logits)` xấp xỉ phân phối đều $1/n\_actions$ (entropy khởi tạo đạt ~`1.943` gần cực đại $\ln(7) \approx 1.946$, độ lệch xác suất cực đại `0.00295 < 0.05`). Điều này đảm bảo Mario khám phá đồng đều mọi hành động ở giai đoạn đầu thay vì sớm kẹt vào một tổ hợp phím.
- **`gain = 1.0` (cho `self.value`):** lớp đầu ra tuyến tính (không qua `ReLU`) thực hiện hồi quy giá trị trạng thái, dùng `gain = 1.0` giúp bảo toàn thang phương sai tự nhiên cho mục tiêu học của Critic.

## Số liệu tham số, bộ nhớ và tốc độ suy luận

Phục vụ đối chiếu tài nguyên huấn luyện và Mục 2 Báo cáo (đo trên Windows với PyTorch 2.6.0+cu124):

- **Tham số `ActorCritic(7)`:** `1,688,232` tham số (**6.44 MiB** FP32), trong đó `trunk = 1,684,128` (99.76%), `policy = 3,591` (0.21%), `value = 513` (0.03%).
- **RAM bộ đệm rollout `(4096, 4, 84, 84)`** (cấu hình mặc định `num_envs=16 × num_steps=256`):
  - Dạng `uint8`: **110.25 MiB** (`115,605,504` bytes).
  - Dạng `float32`: **441.00 MiB** (`462,422,016` bytes).
  - Lưu buffer dạng `uint8` và chỉ chuẩn hóa theo mini-batch trong `forward()` tiết kiệm **4.0× RAM (330.75 MiB)** cho mỗi rollout.

| Thiết bị | Batch size | `act()` (ms / fps) | `forward()` (ms / fps) |
| --- | ---: | ---: | ---: |
| CPU | 1 | `1.211 ms` (`825.6 fps`) | `0.890 ms` (`1,124.2 fps`) |
| CPU | 16 | `2.254 ms` (`7,099.4 fps`) | `8.781 ms` (`1,822.2 fps`) |
| CPU | 64 | `15.635 ms` (`4,093.4 fps`) | `19.103 ms` (`3,350.2 fps`) |
| CPU | 256 | `44.073 ms` (`5,808.5 fps`) | `42.880 ms` (`5,970.2 fps`) |
| CPU | 512 | `62.631 ms` (`8,174.8 fps`) | `54.594 ms` (`9,378.3 fps`) |
| CUDA | 1 | `2.389 ms` (`418.6 fps`) | `0.999 ms` (`1,001.1 fps`) |
| CUDA | 16 | `2.428 ms` (`6,590.0 fps`) | `1.412 ms` (`11,334.0 fps`) |
| CUDA | 64 | `2.482 ms` (`25,782.1 fps`) | `0.806 ms` (`79,413.9 fps`) |
| CUDA | 256 | `4.542 ms` (`56,360.9 fps`) | `1.333 ms` (`192,108.5 fps`) |
| CUDA | 512 | `6.901 ms` (`74,196.4 fps`) | `3.232 ms` (`158,421.4 fps`) |

## Hướng dẫn giao tiếp cho Người 3 và Người 4

- **Cho Người 3 ([`mario_ppo/ppo.py`](../mario_ppo/ppo.py#L230)):**
  - Trong `ppo_update`, truyền trực tiếp `mb["obs"]` (`uint8`, `(mb_size, 4, 84, 84)` cùng device với `model`) vào `logits, values = model(mb["obs"])`.
  - Tạo phân phối qua `Categorical(logits=logits)` để tính `new_logp` và `entropy`. `values` đã là tensor 1D `(mb_size,)` nên trừ trực tiếp với `mb["returns"]` và `mb["old_values"]` không cần `.squeeze()`.
- **Cho Người 4 ([`mario_ppo/train.py`](../mario_ppo/train.py#L86), [`mario_ppo/evaluate.py`](../mario_ppo/evaluate.py#L66)):**
  - Khởi tạo chuẩn: `ActorCritic(n_actions, in_channels=cfg.env.frame_stack, **cfg_dict.get("model", {})).to(device)`.
  - Thu thập rollout và đánh giá gọi `action, logp, value = model.act(obs)`; kết quả trả về luôn trên CPU, không có gradient, dùng được ngay cho `envs.step(action.numpy())` và `env.step(int(action))`.
  - Lưu/tải trọng số bằng `model.state_dict()` và `model.load_state_dict(ck["model"])` đã xác nhận tương thích hoàn toàn.

## Kiểm thử bàn giao

Chạy tại thư mục gốc dự án:

```bash
python -m pytest -v tests/test_model.py
python -m pytest -q tests/test_wrappers.py tests/test_gae.py tests/test_ppo.py
```

- **`tests/test_model.py`:** **5/5 PASS** tuyệt đối (`test_forward_shapes`, `test_forward_accepts_other_action_counts_and_channels`, `test_act_samples_valid_actions_with_matching_logp_and_value`, `test_initial_policy_is_close_to_uniform`, `test_gradients_reach_every_parameter`). [Bài kiểm thử model](../tests/test_model.py)
- **Kiểm thử toàn dự án trên `.venv` Python 3.11 (`torch-2.6.0+cu124`, `mo-gymnasium==1.3.2`, `gym-super-mario-bros==7.4.0`, `nes-py==8.2.1`, `numpy==1.26.4`):** **37/39 PASS** gồm `test_wrappers.py` (**11/11**), `test_gae.py` (**8/8**), `test_ppo.py` (**10/10**), `test_model.py` (**5/5**), và **3/5** bài trong `test_pipeline.py` (gồm `test_make_env_observation_and_reward`, `test_eval_env_has_raw_reward_and_no_noops` trên trình giả lập NES thật).
- **Kiểm thử tích hợp đầu-cuối trên trình giả lập thật (`algo.num_envs=1`):** chạy thông suốt vòng `make_env("1-1")` $\to$ `ActorCritic.act()` $\to$ `compute_gae()` $\to$ `ppo_update()` $\to$ lưu/tải checkpoint $\to$ `evaluate_checkpoint()` trên GPU CUDA không lỗi.

## Điểm chờ đồng bộ từ thành viên khác

1. **Tương thích `forkserver` trên Windows ([`mario_ppo/train.py`](../mario_ppo/train.py#L33) — Người 4):** 2 bài test còn lại trong [`tests/test_pipeline.py`](../tests/test_pipeline.py) (`test_train_smoke_writes_progress_and_checkpoints`, `test_evaluate_checkpoint_reports_flag_rate` dùng `num_envs=2`) báo `ValueError: cannot find context for 'forkserver'` trên Windows vì `AsyncVectorEnv(..., context="forkserver")` chỉ tồn tại trên POSIX (Linux/macOS). Trên Ubuntu CI / Colab / Kaggle cả 39/39 test sẽ chạy qua, hoặc Người 4 có thể bổ sung fallback `"spawn"` cho Windows nếu nhóm huấn luyện song song trên Windows.
2. **Chốt merge Mốc Tuần 1 (Người 4 / Lead):** sau khi PR `feature/n2-actor-critic` và nhánh `feature/n4-ngay1-2-setup` được merge vào `main`, toàn bộ 39/39 test sẽ đạt 100% trên CI để chuyển sang giai đoạn huấn luyện Màn 1-1.
