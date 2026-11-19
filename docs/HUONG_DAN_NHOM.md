# Hướng dẫn chung cho nhóm (Người 4 soạn)

## 1. Cài môi trường

Cần **Python 3.11** (cũng chạy được 3.10/3.12, **không** dùng 3.13 vì `numpy<2` không có bản cho 3.13).

**Windows**: cần cài **Microsoft C++ Build Tools** để biên dịch `nes-py` (chỉ làm một lần, khoảng 2–6 GB):

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Mở terminal mới, rồi tại gốc dự án:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1        # CPU
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gpu   # có GPU NVIDIA
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS / WSL / Colab**: `bash scripts/setup.sh` (Ubuntu cần `sudo apt install build-essential` trước).

Lúc đầu `pytest -q` báo nhiều lỗi `NotImplementedError`. Đó là các phần TODO, không phải lỗi cài đặt.

## 2. Phân nhánh Git

Mỗi người làm trên nhánh riêng (đặt tên tự do), tách từ nhánh tích hợp hiện tại, và mở PR để Người 4 review/merge.

| Người | Tệp chính | Bài test phải PASS |
|---|---|---|
| 1 | `mario_ppo/wrappers.py`, `mario_ppo/env.py` (mẹo) | `tests/test_wrappers.py` (7 bài) |
| 2 | `mario_ppo/model.py` | `tests/test_model.py` (5 bài) |
| 3 | `mario_ppo/ppo.py` | `tests/test_gae.py` (8 bài), `tests/test_ppo.py` (10 bài) |
| 4 | `train.py`, `configs/`, `scripts/` | `tests/test_pipeline.py` (5 bài), toàn bộ `pytest -q` (**35 bài**) |

Quy tắc:
- Chỉ sửa tệp mình sở hữu. Cần sửa tệp của người khác thì báo trong nhóm trước.
- Trước khi mở PR: `pytest -q tests/<test của mình>` phải PASS.
- **Không sửa** `mario_ppo/evaluate.py` (script chấm điểm) và `tests/` (bài kiểm tra công khai).
- Không commit `.venv/`, `runs/`, `*.pt`, video. `.gitignore` đã chặn sẵn. Checkpoint chia sẻ qua Drive.

Thứ tự làm gợi ý (theo README): `test_wrappers` → `test_model` → `test_gae` → `test_ppo` → `test_pipeline`.
Ba phần của Người 1, 2, 3 **độc lập** với nhau (mỗi test chỉ gọi đúng tệp của mình), nên làm song song được.

## 3. Các script tự động (thư mục `scripts/`)

| Script | Việc |
|---|---|
| `setup.ps1` / `setup.sh` | Tạo `.venv`, cài thư viện, kiểm tra import, chạy pytest |
| `train.ps1` | Huấn luyện: đặt tên run, tự chọn `num_envs` theo CPU, ghi `train.log`, in tóm tắt khi xong |
| `smoke_test.ps1` | `pytest -q` rồi huấn luyện 10k bước và in SPS (mốc ngày 5) |
| `evaluate.ps1` | Đánh giá chính thức (seed 2026, 20 lượt) và xuất `ket-qua-<màn>.json` + `<màn>.mp4` |
| `archive_run.ps1` | Chép checkpoint tốt nhất + CSV + config + video vào `results/<màn>/<run>/`, có thể nén `.zip` |
| `run_summary.py` | In SPS, flag_rate, checkpoint tốt nhất, thời gian còn lại, **cảnh báo** (kẹt ống nước, KL lớn, entropy sụp) |
| `plot_progress.py` | Vẽ 9 biểu đồ từ `progress.csv`/`evals.csv`; so sánh nhiều run; `--dpi 300` cho báo cáo |
| `check_interfaces.py` | Kiểm tra các phần của Người 1/2/3 ghép được với `train.py` (shape, dtype, khóa trả về). Dùng khi review PR |

Ví dụ:

```powershell
.\scripts\smoke_test.ps1
.\scripts\train.ps1 -Name run1                                       # 1-1, 5M bước
.\scripts\train.ps1 -Name run1 -Continue                             # máy tắt giữa chừng -> chạy tiếp
.\scripts\train.ps1 -Name run2 -Extra env.max_no_progress_steps=200  # bật mẹo của Người 1
python scripts\run_summary.py runs\1-1\run1                          # xem tiến độ
.\scripts\evaluate.ps1 -Run runs/1-1/run1                            # tự lấy checkpoint tốt nhất
.\scripts\archive_run.ps1 -Run runs/1-1/run1 -Zip
.\scripts\train.ps1 -Level 1-2 -Name transfer -Steps 5000000 -Resume runs/1-1/run1/checkpoints/step_003500000.pt
```

## 4. Ghi chú khi khảo sát `configs/default.yaml`

| Khóa | Giá trị | Ý nghĩa / lưu ý |
|---|---|---|
| `env.actions` | `simple` | 7 hành động (SIMPLE_MOVEMENT); `test_pipeline` kiểm tra `action_space.n == 7` |
| `env.frame_skip`, `frame_stack`, `size` | 4, 4, 84 | Quan sát `(4, 84, 84)` kiểu `uint8`; **phải giống hệt** khi đánh giá |
| `env.weights` | `[1, 1, 2, 0.05, 0.05]` | Trọng số gộp phần thưởng đa mục tiêu `[x_pos, time, death, coin, enemy]` |
| `env.noop_max` | 30 | NOOP ngẫu nhiên đầu lượt khi train (khớp giao thức chấm 0..30 NOOP) |
| `env.normalize_reward` | true | Chỉ khi train, tắt khi đánh giá |
| `algo.num_envs × num_steps` | 16 × 256 = 4096 | Mẫu mỗi lần cập nhật; minibatch = 4096 / 8 = 512 |
| `algo.update_epochs` | 4 | Số vòng quét dữ liệu mỗi lần cập nhật |
| `algo.total_steps` | 5M | Khoảng 1220 lần cập nhật |
| `algo.lr`, `anneal_lr` | 2.5e-4, true | Giảm tuyến tính về 0 (`linear_schedule`) |
| `algo.gamma` | **0.9** | Mặc định của đề là 0.9 (không phải 0.99). Thí nghiệm đối chứng γ dùng khóa này |
| `algo.gae_lambda`, `clip_coef` | 0.95, 0.2 | Chuẩn PPO |
| `algo.ent_coef`, `vf_coef` | 0.01, 0.25 | |
| `algo.target_kl` | null | Không dừng sớm theo KL |
| `checkpoint_every` | 500k | Mỗi checkpoint chạy 5 lượt đánh giá (`eval_episodes`) và ghi video |
| `torch_threads` | 2 | Số luồng torch của tiến trình chính |

**Chọn `num_envs` theo máy** (`train.ps1` tự chọn): ≥16 luồng CPU → 16, 8–15 → 8, ít hơn → 4.
Mỗi môi trường là một tiến trình riêng (Windows dùng `spawn`).

Số đo trên máy Người 4 (16 nhân / 22 luồng, 15.5 GB RAM), chỉ chạy trình giả lập thô, chưa có mạng nơ-ron:

| num_envs | RAM các worker | Khung NES/s | Tương đương SPS (frame_skip=4) |
|---|---|---|---|
| 8 | ~450 MB (~57 MB/worker) | ~1630 | ~410 |
| 16 | ~900 MB | ~2330 | ~580 |

RAM không phải là giới hạn.

**SPS khi huấn luyện đầy đủ** (đo bằng bản cài tạm của các phần TODO, mạng CNN kiểu Nature khoảng 1.7M tham số, CPU, 16 môi trường). Thời gian cho mỗi lần cập nhật 4096 bước:

| `torch_threads` | Thu dữ liệu | Mạng chọn hành động | Cập nhật PPO | SPS thực tế | 5M bước mất |
|---|---|---|---|---|---|
| 2 (mặc định config) | ~7 s | 3.3 s | **36.6 s** | ~110 | ~13 giờ |
| 8 | ~7 s | 1.7 s | 14.4 s | ~150–185 | ~8–9 giờ |
| 12 | | 2.1 s | 17.7 s | (chậm hơn 8) | |

Cập nhật PPO trên CPU là nút thắt. `train.ps1` tự đặt `torch_threads=8` trên máy từ 16 luồng trở lên (4 trên máy 8–15 luồng).
Nếu Người 2 chọn mạng lớn hơn (ví dụ IMPALA-CNN), SPS sẽ giảm thêm, nên đo lại bằng `smoke_test.ps1`.
Máy có GPU NVIDIA nên dùng `setup.ps1 -Gpu`. Khi đó `device: auto` tự chọn CUDA.

**Đã sửa cho Windows**: `train.py` trước đây cố định `context="forkserver"`, mà Windows không có chế độ này nên sẽ crash khi `num_envs > 1`. Giờ code tự chọn `forkserver` nếu hệ điều hành có, nếu không thì dùng `spawn`.
