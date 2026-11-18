# SỔ TAY KỸ THUẬT & PHÂN CÔNG TÁC CHIẾN 14 NGÀY - NGƯỜI 4
## VAI TRÒ: KỸ SƯ TRƯỞNG TÍCH HỢP, HUẤN LUYỆN, ĐÁNH GIÁ & ĐIỀU PHỐI DỰ ÁN (INTEGRATION, TRAINING, BENCHMARK & LEAD)
### Đồ án: Dạy Super Mario Bros vượt màn bằng PPO (PyTorch) - Học phần Học tăng cường

---

## I. THÔNG TIN CHUNG & TRÁCH NHIỆM CỐT LÕI

- **Họ và tên thành viên:** [Điền tên Thành viên 4]
- **Vai trò chuyên trách:** Kỹ sư trưởng Tích hợp hệ thống, Quản trị Huấn luyện, Đánh giá kiểm chuẩn & Trưởng nhóm Học thuật.
- **Tệp mã nguồn sở hữu chính:** `mario_ppo/train.py`, `mario_ppo/evaluate.py`, `mario_ppo/env.py`, `configs/default.yaml`.
- **Bộ bài kiểm tra đơn vị cá nhân:** `tests/test_pipeline.py` (5 bài test) và toàn bộ `pytest -q` (30/30 bài test).
- **Mục tiêu điểm số đóng góp:** Chịu trách nhiệm về sự vận hành trơn tru của toàn bộ hệ thống; đạt trọn vẹn 25/25 điểm Màn 1-1 (Hạng mục B), 15/15 điểm Màn thưởng (Hạng mục C), tổng biên tập Báo cáo 8 trang (25 điểm) và Slide/Demo (5 điểm).

---

## II. ĐẶC TẢ KỸ THUẬT CÁC PHÂN HỆ NGƯỜI 4 PHỤ TRÁCH

Người 4 tuyệt đối **KHÔNG CHỈ LÀM QUẢN LÝ HÀNH CHÍNH** mà nắm giữ các mắt xích kỹ thuật sống còn của dự án:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      KIẾN TRÚC HỆ THỐNG DO NGƯỜI 4 QUẢN TRỊ            │
│                                                                        │
│ 1. Vector Environment: AsyncVectorEnv(num_envs=16, context="forkserver")│
│    - Tự động hóa reset: AutoresetMode.SAME_STEP                        │
│    - Bóc tách final_obs khi bị cắt ngắn (truncated=1)                  │
│                                                                        │
│ 2. Vòng lặp Rollout (train.py): T=256 bước trên N=16 môi trường        │
│    - Bộ đệm tensor uint8 trên CPU: b_obs, b_act, b_logp, b_val, b_rew  │
│    - Gọi GAE từ Người 3: compute_gae(...) -> advantages, returns       │
│    - Gọi PPO Update từ Người 3: ppo_update(model, opt, batch, a)       │
│                                                                        │
│ 3. Giám sát & Nhật ký:                                                 │
│    - Ghi progress.csv (sps, lr, loss, return, x_pos, flag_rate)        │
│    - Lưu Checkpoint mỗi 500k bước: checkpoints/step_*.pt                │
│    - Tự động đánh giá 5 lượt và xuất video: evals.csv & videos/*.mp4   │
│                                                                        │
│ 4. Đánh giá chuẩn mực (evaluate.py):                                   │
│    - Cố định seed 2026, 20 lượt xuất phát trễ 0..30 NOOP               │
│    - Xuất ket-qua-*.json và video mp4 lượt tốt nhất                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Quản lý Hạ tầng Huấn luyện Song song trong `mario_ppo/train.py`

Người 4 làm chủ cơ chế đa tiến trình trong `_make_vec_env`:
```python
def _make_vec_env(env_cfg, level, num_envs, seed):
    from mario_ppo.env import make_env
    fns = [functools.partial(make_env, level, seed=seed + i, **env_cfg) for i in range(num_envs)]
    mode = gym.vector.AutoresetMode.SAME_STEP
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)
    return gym.vector.AsyncVectorEnv(fns, context="forkserver", autoreset_mode=mode)
```

- **Quy tắc an toàn RAM:** Không import PyTorch ở mức module trong các tiến trình con (tiết kiệm ~300 MB RAM cho mỗi môi trường).
- **Cân chỉnh tài nguyên theo số nhân CPU:**
  - Máy 16 luồng (Core i7 / Ryzen 7): `algo.num_envs=16`, `torch_threads=2`.
  - Máy 8 luồng (Laptop 4-6 cores): `algo.num_envs=8`, `torch_threads=2`.
- **Cơ chế phục hồi an toàn (Resume):**
  Khi tiến trình bị dừng đột ngột, Người 4 chạy tiếp bằng lệnh:
  ```powershell
  python -m mario_ppo.train resume=runs/1-1/<run>/checkpoints/step_002500000.pt
  ```
  Tiến trình sẽ khôi phục trọng số mạng, bộ nhớ Adam, `global_step` và tiếp tục chạy đến `total_steps`.

---

### 2. Quản lý Giao thức Đánh giá Chuẩn mực trong `mario_ppo/evaluate.py`

Người 4 thực thi script đánh giá chính thức với hạt giống công khai `seed=2026` theo đúng quy định tại Mục 4 đề bài:

```powershell
# 1. Đánh giá Màn 1-1 (Hạng mục B - 25 điểm)
python -m mario_ppo.evaluate --ckpt runs/1-1/best/checkpoints/step_003500000.pt --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4

# 2. Đánh giá Màn thưởng 1-2 (Hạng mục C - 15 điểm)
python -m mario_ppo.evaluate --ckpt runs/1-2/best/checkpoints/step_004500000.pt --level 1-2 --episodes 20 --seed 2026 --out ket-qua-1-2.json --video 1-2.mp4
```

- **Tiêu chuẩn vượt màn:** Tỷ lệ chạm cờ đạt `flag_rate >= 0.8` (chạm cờ ít nhất 16/20 lượt).
- **Tệp đầu ra bắt buộc:** `ket-qua-1-1.json` (ghi nhận `solved: true`, `mean_x_pos`, danh sách 20 lượt) và `1-1.mp4` (video độ nét cao của lượt chạm cờ đẹp nhất).

---

### 3. Chiến lược Huấn luyện Màn thưởng: Transfer Learning

Người 4 chỉ đạo chiến dịch Màn thưởng 1-2 bằng phương pháp Học chuyển giao (Transfer Learning) để tiết kiệm 60% thời gian tính toán:
```powershell
python -m mario_ppo.train level=1-2 resume=runs/1-1/best/checkpoints/step_003500000.pt run_dir=runs/1-2/transfer_run algo.total_steps=5000000
```
Tận dụng các bộ lọc CNN trích xuất đặc trưng của Màn 1-1, Mario sẽ nhanh chóng thích nghi với địa hình hầm ngầm của Màn 1-2 chỉ trong vòng 3 – 5 triệu bước tác tử (khoảng 3 – 4 giờ chạy trên máy đa lõi).

---

## III. QUY TRÌNH KIỂM THỬ VÀ LỆNH CHẠY CỦA NGƯỜI 4

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 1. Kiểm tra tích hợp pipeline
pytest -q tests/test_pipeline.py

# 2. Kiểm tra toàn bộ 30 bài test công khai của dự án (BẮT BUỘC PASS 100%)
pytest -q

# 3. Chạy thử nghiệm huấn luyện ngắn (Smoke test 10,000 bước)
python -m mario_ppo.train algo.total_steps=10000 run_dir=runs/smoke_test

# 4. Kiểm tra cấu hình và đo tốc độ SPS
python -c "import pandas as pd; df = pd.read_csv('runs/smoke_test/progress.csv'); print('Tốc độ trung bình:', df['sps'].mean(), 'SPS')"
```

---

## IV. LỘ TRÌNH 14 NGÀY CHI TIẾT CỦA NGƯỜI 4

- **Ngày 1:** Khởi tạo Git repo, phân nhánh tính năng, hướng dẫn cả nhóm cài đặt môi trường, chạy thử `pytest -q` kiểm tra các lỗi TODO.
- **Ngày 2:** Khảo sát tệp `configs/default.yaml`, viết các script bash/powershell tự động hóa quy trình chạy huấn luyện và lưu trữ.
- **Ngày 3:** Rà soát code của Người 1, 2, 3 trên các nhánh feature; kiểm tra giao diện tensor đầu vào/đầu ra.
- **Ngày 4:** Merge nhánh integration; giải quyết các xung đột merge conflict; sửa lỗi mismatch tensor giữa CNN và PPO.
- **Ngày 5 (CỘT MỐC TUẦN 1):** Chạy `pytest -q tests/test_pipeline.py` (5/5 PASS) và `pytest -q` đạt **30/30 PASS tuyệt đối** (30 điểm Hạng mục A). Chạy smoke test 10k bước và đo SPS ban đầu.
- **Ngày 6:** Khởi chạy Run 1 chính thức cho Màn 1-1 (5M bước) trên máy trạm đa lõi mạnh nhất. Đo đạc SPS thực tế.
- **Ngày 7:** Đọc nhật ký `progress.csv`. Nếu Mario kẹt ở ống nước đầu tiên ($x < 400$), phối hợp Người 1 bật mẹo `NoProgressTruncation` và chạy Run 2 song song.
- **Ngày 8:** Quá trình huấn luyện Màn 1-1 cán mốc 3.5M–5M bước. Lọc các checkpoint có `flag_rate` cao trong `evals.csv`.
- **Ngày 9 (CỘT MỐC MÀN 1-1):** Chạy lệnh đánh giá chuẩn:
  `python -m mario_ppo.evaluate --ckpt runs/1-1/best/checkpoints/step_003500000.pt --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4`.
  Xác nhận `flag_rate >= 0.8` (25 điểm Hạng mục B). Khởi chạy ngay Transfer Learning cho Màn thưởng 1-2.
- **Ngày 10:** Quản trị tiến trình huấn luyện Màn thưởng 1-2; hỗ trợ Người 3 chạy Thí nghiệm đối chứng $\gamma=0.9$ vs $\gamma=0.99$ trên Màn 1-1 (2M bước).
- **Ngày 11:** Màn thưởng cán mốc huấn luyện; thu thập số liệu Thí nghiệm so sánh; phân tích biểu đồ phân bố điểm chết.
- **Ngày 12 (CỘT MỐC MÀN THƯỞNG):** Đánh giá chuẩn Màn thưởng 1-2 với seed 2026 đạt `flag_rate >= 0.8` (15 điểm Hạng mục C). Xuất `ket-qua-1-2.json` và `1-2.mp4`. Xuất 7 biểu đồ chuẩn 300 DPI.
- **Ngày 13:** Đóng băng mã nguồn (tag `v1.0-final`). Tổng biên tập Báo cáo học thuật 6 mục; khóa file PDF đúng tối đa 8 trang; thiết kế Slide 10 phút.
- **Ngày 14:** Tổ chức 2 buổi tổng duyệt thuyết trình và phản biện vấn đáp; kiểm tra tính tái lập trên máy tính độc lập; đóng gói hồ sơ `nhom-<so>.zip` nộp bài.

---

## V. BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO NGƯỜI 4

1. **Hỏi:** *Tại sao trong train.py lại dùng gym.vector.AsyncVectorEnv với context='forkserver'?*
   - **Đáp:** Môi trường NES chạy đơn luồng CPU. Để thu dữ liệu nhanh, ta phải chạy song song N môi trường (ví dụ N=16). Trên Linux/macOS, fork mặc định có thể gây deadlock khi kết hợp với PyTorch đa luồng. Context 'forkserver' khởi tạo một tiến trình server sạch và fork các worker môi trường từ đó, loại bỏ nguy cơ deadlock và tối đa hóa chỉ số SPS.
2. **Hỏi:** *Cơ chế AutoresetMode.SAME_STEP trong gym.vector vận hành như thế nào?*
   - **Đáp:** Trong SAME_STEP, khi một môi trường con kết thúc lượt (`done=True`), nó tự động `reset()` ngay trong chính bước thời gian đó và trả về quan sát đầu tiên của lượt mới trong `next_obs`. Tuy nhiên, để tính GAE chính xác, ta bắt buộc phải biết quan sát cuối cùng của lượt vừa kết thúc (`final_obs`), được wrapper lưu trong `info['final_obs']`. Nhờ đó, train.py có thể lấy giá trị bootstrap `v_fin` chính xác cho bước bị cắt ngắn.
3. **Hỏi:** *Tại sao lại chọn T = 256 bước rollout và N = 16 môi trường (tổng 4096 mẫu/cập nhật)?*
   - **Đáp:** 4096 mẫu là kích thước lô chuẩn mực cho Mario và Atari. N=16 môi trường đảm bảo dữ liệu có độ đa dạng cao, phá vỡ tính tương quan chuỗi thời gian. T=256 bước tác tử (tương đương 1024 khung NES do skip=4) đủ dài để bao quát một chuỗi hành vi hoàn chỉnh (lấy đà, nhảy qua ống nước, tiếp đất an toàn), giúp GAE ước lượng chính xác giá trị chuỗi hành động.
4. **Hỏi:** *Ý nghĩa của việc giảm tuyến tính tốc độ học (Anneal Learning Rate)?*
   - **Đáp:** Hàm `linear_schedule` giảm tốc độ học đều đặn từ $2.5 \times 10^{-4}$ về $0$. Ban đầu, lr lớn giúp mạng nhanh chóng thoát khỏi vùng khởi tạo ngẫu nhiên; ở giai đoạn cuối khi chính sách đã gần tối ưu, lr nhỏ giúp các trọng số hội tụ mịn vào điểm cực tiểu, tránh bước nhảy quá lớn làm hỏng chính sách đã học.
5. **Hỏi:** *Tại sao khi đánh giá bắt buộc phải dùng seed 2026 và chạy đúng 20 lượt?*
   - **Đáp:** Seed 2026 sinh ra một chuỗi 20 số nguyên cố định $n_i \in \{0, \dots, 30\}$ đại diện cho số bước NOOP Mario đứng yên lúc bắt đầu mỗi lượt. Do chu kỳ xuất hiện của kẻ thù trong NES phụ thuộc vào thời điểm di chuyển, việc cố định seed 2026 đảm bảo tất cả các nhóm sinh viên trên toàn khóa đều gặp đúng 20 tình huống xuất phát giống hệt nhau. Đánh giá 20 lượt loại bỏ tính may rủi ngẫu nhiên, chứng minh chính sách có tính tổng quát hóa bền vững.

---

## VI. CHECKLIST NGHIỆM THU CỦA NGƯỜI 4

- [ ] Chạy `pytest -q tests/test_pipeline.py` đạt **5/5 PASS**.
- [ ] Chạy toàn bộ `pytest -q` đạt **30/30 PASS tuyệt đối** (Đạt trọn vẹn 30 điểm Hạng mục A).
- [ ] Huấn luyện Màn 1-1 hoàn tất 3.5M–5M bước; có `progress.csv` và `evals.csv`.
- [ ] Đánh giá Màn 1-1 qua `evaluate.py` (seed 2026, 20 lượt) đạt `flag_rate >= 0.8` (25 điểm Hạng mục B).
- [ ] Xuất tệp `ket-qua-1-1.json` và video `1-1.mp4`.
- [ ] Huấn luyện Transfer Learning Màn thưởng 1-2 đạt `flag_rate >= 0.8` (15 điểm Hạng mục C).
- [ ] Xuất tệp `ket-qua-1-2.json` và video `1-2.mp4`.
- [ ] Hoàn thành Thí nghiệm đối chứng $\gamma=0.9$ vs $\gamma=0.99$ cùng Người 3.
- [ ] Xuất bản 7 biểu đồ phân tích chuẩn 300 DPI.
- [ ] Khóa thành công Báo cáo học thuật PDF đúng tối đa 8 trang (Hạng mục D - 25 điểm).
- [ ] Biên tập bộ Slide 10 phút và tổ chức 2 buổi tổng duyệt vấn đáp (Hạng mục E - 5 điểm).
- [ ] Đóng gói tệp nộp bài `nhom-<so>.zip` đầy đủ 8 thành phần theo Mục 7 đề bài.
- [ ] Kiểm tra tính tái lập trên máy độc lập thành công.
