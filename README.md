# GIẢI QUYẾT TRÒ CHƠI SUPER MARIO BROS (WORLD 1-1) BẰNG DEEP REINFORCEMENT LEARNING

> **Đồ án cuối kỳ môn học: Học tăng cường (Reinforcement Learning)**  
> **Quy mô nhóm:** 04 thành viên | **Thời gian thực hiện:** 05 tuần (35 ngày)  
> **Mục tiêu điểm số:** 10.0 / 10.0 (8.0đ Mục tiêu chính + 2.0đ Cải tiến mở rộng)

## Phần môi trường đã triển khai

Yêu cầu **Python 3.13 64-bit**. Môi trường dùng Mario 9.1.0, nes-py 9.0.1
và Gymnasium 1.2.3; bộ phiên bản cũ trong các sổ tay là tài liệu kế hoạch.

Chạy từ thư mục gốc repository trên PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe baseline_random.py --steps 10000
```

Trên Linux, dùng `.venv/bin/python` thay cho `.venv\Scripts\python.exe`.
`requirements.txt` chứa phụ thuộc runtime; `requirements-dev.txt` thêm pytest.

```python
from src.env import make_mario_env

with make_mario_env() as env:
    obs, info = env.reset(seed=42)  # obs: uint8 (4, 84, 84)
    obs, reward, terminated, truncated, info = env.step(1)
    if terminated or truncated:
        obs, info = env.reset()
```

Pipeline: Joypad → lặp tối đa 4 frame + max-pool → giới hạn bước agent →
reward hook tùy chọn → grayscale/resize → stack 4 ảnh → thống kê episode.
Hỗ trợ SIMPLE_MOVEMENT (7 actions), RIGHT_ONLY (5 actions), render RGB/human.

Có thêm `RIGHT_JUMP` gồm 2 hành động: `0` đi phải, `1` đi phải + nhảy.
Dùng `make_mario_env(action_type="RIGHT_JUMP")` hoặc chạy:

```powershell
.\.venv\Scripts\python.exe baseline_random.py --action-type RIGHT_JUMP --steps 1000
```

Cấu hình riêng: `configs/env_two_actions.json`. Cấu hình mặc định vẫn là
`SIMPLE_MOVEMENT`. Khi tạo mạng, lấy số action từ `env.action_space.n`.

Có thể mở game bằng `python baseline_random.py --render --steps 1000` sau khi
kích hoạt venv. Thuật toán train, reward riêng và đánh giá mô hình sẽ được các
phần tiếp theo bổ sung.

## Tài liệu bàn giao môi trường

**File bàn giao cho Người 2, 3 và 4:** [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — hướng dẫn API, seed, đơn vị frame/bước, cách gắn reward và xuất ảnh.

## Cấu trúc repository

```text
RL_Mario_Solving/
├── configs/
│   ├── env.json             # Cấu hình môi trường mặc định (7 hành động)
│   └── env_two_actions.json # Cấu hình 2 hành động
├── src/
│   ├── env/                 # Factory và wrappers đã triển khai
│   ├── agents/              # Nơi Người 2 bổ sung thuật toán
│   ├── models/              # Nơi Người 2 bổ sung mạng
│   ├── rewards/             # Nơi Người 3 bổ sung reward shaping
│   └── utils/               # Tiện ích chung
├── tests/                   # Kiểm thử wrapper và game thật
├── docs/ENVIRONMENT.md      # Hợp đồng API và hướng dẫn bàn giao
├── checkpoints/             # Trọng số sau này
├── recordings/              # Ảnh/video đầu ra
├── logs/                    # Log huấn luyện sau này
├── results/                 # JSON benchmark sinh tự động
├── baseline_random.py       # Chạy random agent, đo tốc độ/RAM
├── requirements.txt
└── requirements-dev.txt
```

---

## 📌 HỆ THỐNG TÀI LIỆU KẾ HOẠCH & SỔ TAY KỸ THUẬT (MARKDOWN)

Toàn bộ kế hoạch triển khai, phân công chi tiết và mã nguồn mẫu được lưu trữ tập trung tại thư mục: [`Phan_Cong_Nhiem_Vu/`](Phan_Cong_Nhiem_Vu/)

1. 📘 [`KE_HOACH_TONG_THE_5_TUAN.md`](Phan_Cong_Nhiem_Vu/KE_HOACH_TONG_THE_5_TUAN.md) *(57 KB)*: Kế hoạch tổng thể dự án 5 tuần, mô hình toán học MDP, WBS 26 hạng mục công việc (P0/P1/P2), ma trận RACI, kế hoạch 12 chương báo cáo, kế hoạch slide, quản lý rủi ro và tiêu chí nghiệm thu.
2. 📗 [`NGUOI_1_MOI_TRUONG_VA_TIEN_XU_LY.md`](Phan_Cong_Nhiem_Vu/NGUOI_1_MOI_TRUONG_VA_TIEN_XU_LY.md) *(17 KB)*: Sổ tay kỹ thuật Người 1 (Môi trường Gymnasium, Custom Wrappers: Grayscale, Resize $84 \times 84$, FrameSkip = 4, FrameStack = 4, JoypadSpace, code template `make_mario_env()`, xử lý lỗi `nes-py`, ngân hàng câu hỏi vấn đáp riêng).
3. 📕 [`NGUOI_2_THUAT_TOAN_DQN_PPO.md`](Phan_Cong_Nhiem_Vu/NGUOI_2_THUAT_TOAN_DQN_PPO.md) *(21 KB)*: Sổ tay kỹ thuật Người 2 (Mạng PyTorch Nature CNN + Dueling Head, Double Q-learning, Replay Buffer lưu mảng `uint8` trên RAM, Huber Loss, code template `DoubleDQNAgent`, thuật toán đối sánh PPO).
4. 📙 [`NGUOI_3_HUAN_LUYEN_REWARD_THUC_NGHIEM.md`](Phan_Cong_Nhiem_Vu/NGUOI_3_HUAN_LUYEN_REWARD_THUC_NGHIEM.md) *(16 KB)*: Sổ tay kỹ thuật Người 3 (Lý thuyết Reward Shaping, khắc phục 4 bẫy hành vi: sợ quái vật nhảy lùi, kẹt ống nước cao; code template `MarioCustomRewardWrapper` v2 tích hợp Anti-stuck penalty; TensorBoard metrics; kịch bản Ablation Study 2 điểm).
5. 📓 [`NGUOI_4_TICH_HOP_TEST_BAO_CAO_DEMO.md`](Phan_Cong_Nhiem_Vu/NGUOI_4_TICH_HOP_TEST_BAO_CAO_DEMO.md) *(17 KB)*: Sổ tay kỹ thuật Người 4 (Git Flow, Pytest suite, code template `evaluate.py` 20 seeds ngẫu nhiên với $\epsilon = 0.0$, code template `demo.py` hiển thị HUD thời gian thực và ghi video MP4 full-HD, kiểm thử tính tái lập, điều phối Báo cáo IEEE và Slide).

> *Lưu ý: Nếu cần xuất file Word (.docx) để in ấn hoặc nộp cho giảng viên, chỉ cần chạy lệnh: `python generate_docx.py`.*

---

## 👥 PHÂN CHIA VAI TRÒ 4 THÀNH VIÊN

| STT | Thành viên | Vai trò chuyên trách | Nhiệm vụ kỹ thuật cốt lõi |
| :---: | :---: | :--- | :--- |
| **1** | **Người 1** | Kỹ sư Môi trường & Tiền xử lý | Môi trường Gymnasium, Custom Wrappers (84x84, Skip 4, Stack 4), Joypad 7 actions, baseline test |
| **2** | **Người 2** | Kỹ sư Thuật toán DRL & Mạng | Mạng PyTorch Dueling CNN, Double DQN, Replay Buffer uint8, đối sánh PPO |
| **3** | **Người 3** | Kỹ sư Huấn luyện & Thực nghiệm | Reward Shaping v2 (Anti-stuck), TensorBoard log, train 3M steps, Ablation Study |
| **4** | **Người 4** | Kỹ sư Tích hợp, QA & Báo cáo | Git Flow, Pytest, evaluate.py (20 seeds), demo.py HUD & MP4, Tổng tài Báo cáo |

---

## 🎯 MỤC TIÊU NGHIỆM THU CHÍNH (8.0 ĐIỂM + 2.0 ĐIỂM)

Các mục dưới là mục tiêu trong kế hoạch học phần, chưa phải kết quả đã đo.

1. **Mục tiêu chính (8 điểm):** Mario hoàn thành màn **World 1-1** (`flag_get == True`, $X \ge 3161$), tỷ lệ thắng $\ge 80\%$ trên 20 test episodes độc lập ($\epsilon = 0.0$).
2. **Cải tiến mở rộng (2 điểm):** Nghiên cứu đối sánh Dueling DDQN vs PPO; Ablation Study về Reward Shaping; Ứng dụng Demo HUD thời gian thực kèm video MP4 full-HD.
